import asyncio
import copy
import importlib.util
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import litellm
import numpy as np
import psutil
import torch
from collections import Counter
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from recipe.usim.logging_utils import as_bool, debug_enabled, debug_print, debug_print_kh, shorten

INVALID_NAME_CHARS_RE = re.compile(r'[\s<|\\/>]')
def is_valid_gpt_name(name: str) -> bool:
    return bool(name) and not INVALID_NAME_CHARS_RE.search(name)

async def get_gpt5_mini_generation(raw_prompt: list, max_retry=3) -> Optional[str]:
    """
    get gen for gpt-5-mini. We try medium reasoning effort max_retry times
    then try low once then minimal once

    raw_prompt is list of messages from create_any_dataset in the format
    [{"role": ..., "content": ..., "name": ... (optional)}, ...]
    max_retry: Number of retries per reasoning effort level
    """
    import openai
    import asyncio

    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.AsyncOpenAI(api_key=api_key)

    input_messages = []
    for msg in raw_prompt:
        role = msg.get("role", "user")

        if role == "system":
            msg.pop("name")

        content = msg.get("content", "")

        m = {"role": role, "content": content}
        name = msg.get("name", None)
        if name is not None:
            if not is_valid_gpt_name(name):
                print(name)
                name = name.replace(" ", "_")
        m["name"] = name

        input_messages.append(m)

    reasoning_efforts = ["medium", "low", "minimal"]
    print('======================')
    print(input_messages)

    for effort in reasoning_efforts:
        if effort == "medium":
            max_retry = max_retry
        else:
            max_retry = 1

        for attempt in range(max_retry):
            try:
                completion = await client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=input_messages,
                    reasoning_effort=effort,
                )

                # Extract text from the response
                output_text = (completion.choices[0].message.content or "")

                if output_text.strip():
                    return output_text.strip()
                else:
                    print(
                        f"[GPT-5-mini] Attempt {attempt+1}/{max_retry} with effort={effort}: Empty generation"
                    )

            except Exception as e:
                print(
                    f"[GPT-5-mini] Attempt {attempt+1}/{max_retry} with effort={effort} failed: {e}"
                )
                if attempt < max_retry - 1:
                    await asyncio.sleep(1)
        print(
            f"[GPT-5-mini] All {max_retry} attempts with effort={effort} failed, trying next effort level..."
        )

    print("[GPT-5-mini] All reasoning effort levels exhausted, returning None")
    return None

METRIC_IMPORT_LOCK = threading.Lock()

async def compute_reward(
    data_source: str,
    generations: list[str],
    ground_truth: str,
    metrics: dict[str, dict[str, Any]],
    extra_info: dict[str, Any] | None = None,
    num_retries: int = 10,
    eval_llm_api: str = None,
) -> dict[str, list[float]]:
    """Compute rewards for generations using specified metrics."""


    async def load_metric(metric: str):
        """Dynamically load metric module with caching."""
        metric_path = Path(__file__).parent / "metrics" / f"{metric}.py"
        if not metric_path.exists():
            raise FileNotFoundError(f"Metric not found: {metric_path}")

        with METRIC_IMPORT_LOCK:
            if metric in sys.modules:
                return sys.modules[metric]

            spec = importlib.util.spec_from_file_location(metric, metric_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not load metric '{metric}'")

            module = importlib.util.module_from_spec(spec)
            sys.modules[metric] = module
            spec.loader.exec_module(module)

        return module

    async def compute_with_retry(fn, is_async: bool, metric_name: str, *args, **kwargs):
        """Execute with exponential backoff retry."""
        for attempt in range(num_retries):
            try:
                return await fn(*args, **kwargs) if is_async else fn(*args, **kwargs)
            except Exception as e:
                # Debug: include metric context + snippet of what we were scoring
                if debug_enabled(extra_info):
                    scored = ""
                    try:
                        # args is typically: (data_source, generation(s), ground_truth, extra_info, ...)
                        gen_arg = args[1] if len(args) > 1 else None
                        if isinstance(gen_arg, list):
                            scored = shorten(gen_arg[0] if gen_arg else "", 400)
                        else:
                            scored = shorten(gen_arg, 400)
                    except Exception:
                        scored = ""
                    debug_print(
                        extra_info,
                        f"metric_exception: metric={metric_name} attempt={attempt+1}/{num_retries} "
                        f"err_type={type(e).__name__} err='{shorten(e, 240)}' scored_prefix='{scored}'",
                    )
                if attempt == num_retries - 1:
                    print(f"[Error] Final failure in metric '{metric_name}': {e}", flush=True)
                    return 0.0
                print(f"[Retry {attempt + 1}] metric='{metric_name}' err={e}", flush=True)
                if isinstance(e, litellm.RateLimitError):
                    await asyncio.sleep(2 ** attempt)

    ################### EVAL GPT/other API LLM ##############
    # replace generations w/ api calls
    # [WARNING] ASSUMING NO BATCH CALLS 
    if eval_llm_api:
        print("============================")
        print("EVAL LLM API IS NONE: ", bool(eval_llm_api is None))
        import pdb; pdb.set_trace()
        import json
        assert len(generations) == 1
        raw_prompt = json.loads(extra_info["raw_prompt"])
        gpt_generation = await get_gpt5_mini_generation(raw_prompt, max_retry=3)
        
        if gpt_generation is None:
            raise ValueError("OUTPUT FROM GPT-5-MINI FAILED")
        
        generations = [gpt_generation]


    hierarchy_name = extra_info.get("hierarchy_name") if extra_info else None
    hierarchy = extra_info.get("hierarchy") if extra_info else None
    debug_print(
        extra_info,
        f"compute_reward: metrics={list(metrics.keys())} num_generations={len(generations)} "
        f"gt_len={len(ground_truth or '')} hierarchy_name={hierarchy_name} hierarchy_len={len(hierarchy or [])}",
    )
    if hierarchy_name == hierarchy[-1]:
        gd_mask = [gen.strip() == ground_truth.strip() for gen in generations]
    else:
        gd_mask = [False] * len(generations)

    # Filter empty generations
    has_content = [bool(gen.strip()) for gen in generations]
    has_content = [hc and (not is_gd) for hc, is_gd in zip(has_content, gd_mask)]
    non_empty = [g for g, keep in zip(generations, has_content) if keep]
    debug_print(
        extra_info,
        f"content_filter: has_content={sum(has_content)}/{len(has_content)} gd_count={sum(gd_mask)} "
        f"sample_generation='{shorten((non_empty[0] if non_empty else generations[0] if generations else ''), 220)}'",
    )
    
    # if not any(has_content):
    #     score_lst = [0.0 if not gd_flag else 1.0 for (gen, gd_flag) in zip(generations, gd_mask)]
    #     reward_dict = {metric: score_lst for metric in metrics}
    #     return reward_dict
    # CHECK
    if not any(has_content):
        score_lst = [0.0 if not gd_flag else 1.0 for gd_flag in gd_mask]
        reward_dict = {}
        for metric, config in metrics.items():
            # For state_reward_on_response, get the sub-metric keys from its config_path
            # we need state_reward_on_response:score, ... for consistency
            if metric == "state_reward_on_response":
                config_path = config.get("kwargs", {}).get("config_path")
                if config_path:
                    import json
                    sub_config = json.load(open(config_path, "r"))
                    sub_keys = list(sub_config.keys())  
                    # ['stance', 'emotion', 'belief', 'value', 'goal', 'communication']
                    for sub_key in sub_keys:
                        reward_dict[f"{metric}:{sub_key}"] = score_lst.copy()
                    reward_dict[f"{metric}:metrics_info"] = [""] * len(score_lst)
                else:
                    reward_dict[f"{metric}:score"] = score_lst.copy()
                    reward_dict[f"{metric}:metrics_info"] = [""] * len(score_lst)
            else:
                reward_dict[f"{metric}:score"] = score_lst.copy()
                reward_dict[f"{metric}:metrics_info"] = [""] * len(score_lst)
        return reward_dict

    # Compute scores for each metric
    reward_dict = {}
    for metric, config in metrics.items():
        kwargs = config.get("kwargs", {})
        
        module = await load_metric(metric)
        # Use batched function if available
        if hasattr(module, "compute_batch_score"):
            fn = module.compute_batch_score
            scores_and_info = await compute_with_retry(
                fn,
                asyncio.iscoroutinefunction(fn),
                metric,
                data_source,
                non_empty,
                ground_truth,
                extra_info,
                **kwargs,
            )
            assert isinstance(scores_and_info, list), f"Expected list from compute_batch_score, got {type(scores_and_info)}"

        elif hasattr(module, "compute_score"):
            fn = module.compute_score
            is_async = asyncio.iscoroutinefunction(fn)
            scores_and_info = [
                await compute_with_retry(fn, is_async, metric, data_source, pred, ground_truth, extra_info, **kwargs)
                for pred in non_empty
            ]
        else:
            print(f"[Error] Metric '{metric}' missing compute function")
            raise ValueError(f"Metric '{metric}' missing compute function")
        
        # Debug: summarize metric output schema + score ranges (excluding metrics_info)
        if debug_enabled(extra_info) and scores_and_info:
            first = scores_and_info[0]
            if isinstance(first, dict):
                keys = list(first.keys())
                numeric_keys = [k for k in keys if k != "metrics_info"]
                ranges = {}
                for k in numeric_keys:
                    try:
                        vals = [float(x.get(k)) for x in scores_and_info if isinstance(x, dict)]
                        ranges[k] = (min(vals), float(np.mean(vals)), max(vals))
                    except Exception:
                        continue
                debug_print(extra_info, f"metric_return: metric={metric} keys={keys} numeric_ranges={ranges}")
            else:
                try:
                    vals = [float(x) for x in scores_and_info]
                    debug_print(extra_info, f"metric_return: metric={metric} scalar_range={(min(vals), float(np.mean(vals)), max(vals))}")
                except Exception:
                    debug_print(extra_info, f"metric_return: metric={metric} type={type(first)}")

        out_keys = scores_and_info[0].keys() if isinstance(scores_and_info[0], dict) else (None,)
        for key in out_keys:
            auto_field = '' if key == 'metrics_info' else 0.0
            full_scores_and_info, idx = [], 0
            for keep in has_content:
                full_scores_and_info.append((scores_and_info[idx][key] if key is not None else scores_and_info[idx]) if keep else auto_field)
                idx += keep
                
            if not key == 'metrics_info':
                gd_score = min(1.0, max(full_scores_and_info) + 0.1)
                for i, is_gd in enumerate(gd_mask):
                    if is_gd:
                        full_scores_and_info[i] = gd_score
                
            if key is not None:
                reward_dict[f"{metric}:{key}"] = full_scores_and_info
            else: 
                reward_dict.setdefault(metric, full_scores_and_info)
    debug_print(extra_info, f"reward_dict_keys={list(reward_dict.keys())}")
    
    return reward_dict


HIERARCHY_TEMPLATE = "<{name}>\n{field}\n</{name}>"
HALF_HIERARCHY_TEMPLATE = "\n{field}\n</{name}>"
FIELD_PATTERN = re.compile(r"<(?P<name>\w+)>(?P<field>.*?)</\1>", re.DOTALL | re.IGNORECASE)

def parse_any_fields(text: str) -> Dict[str, str]:
    """
    Parse <field>content</field> blocks into a dictionary.
    """
    matches = FIELD_PATTERN.findall(text or "")
    return {name.lower().strip(" \n"): field.strip() for name, field in matches}
    

def parse_fields_strict(text: str, hierarchy: List[str]) -> Dict[str, str]:
    """
    Strictly parse a sequence of <field>...</field> blocks:
      - Trim only leading/trailing ' ' and '\n' from the entire input
      - Must START with <hierarchy[0]> exactly (case-insensitive)
      - Each field appears EXACTLY once, in EXACT order
      - Only ' ' and '\n' allowed between blocks
      - No extra content before first or after last block
      - PRESERVE inner content exactly (no stripping)
    Returns {} on any violation.
    """
    if text is None:
        return {}

    s = text.strip(" \n")
    if not s:
        return {}

    # Must start with the first tag exactly after trimming
    first_tag_rx = re.compile(rf"^<{re.escape(hierarchy[0])}>", re.IGNORECASE)
    if not first_tag_rx.search(s):
        return {}

    # Build a single anchored regex enforcing order and exact-once,
    # allowing only spaces/newlines between blocks.
    parts = []
    for i, field in enumerate(hierarchy):
        tag = re.escape(field)
        # Capture inner content EXACTLY as-is (no surrounding \s* in the group)
        block = rf"<{tag}>(?P<g{i}>.*?)</{tag}>"
        parts.append(block)

    # Only spaces/newlines allowed between blocks; anchor ^...$
    between = r"[ \n]*"
    pattern = r"^" + between.join(parts) + r"$"
    rx = re.compile(pattern, re.DOTALL | re.IGNORECASE)

    m = rx.match(s)
    if not m:
        return {}

    # Build result preserving inner content verbatim
    result: Dict[str, str] = {}
    for i, field in enumerate(hierarchy):
        content = m.group(f"g{i}")
        # No strip here — preserve leading/trailing whitespace inside the tag
        result[field.strip()] = "" if content is None else content

    return result
    

def parse_fields_strict_thinking(text: str, hierarchy: List[str]) -> Dict[str, str]:
    """
    Same as above but we look for "[think text] </think>" before
    """
    if text is None:
        return {}

    s = text.strip(" \n")
    if not s:
        return {}

    if 'think' not in hierarchy:
        # Must start with the first tag exactly after trimming
        first_tag_rx = re.compile(rf"^<{re.escape(hierarchy[0])}>", re.IGNORECASE)
        if not first_tag_rx.search(s):
            m_think = re.search(r"(</think>|<\\think>)", s, re.IGNORECASE)
            if not m_think:
                return {}
            s = s[m_think.end():].strip(" \n")
            if not s:
                return {}
            if not first_tag_rx.search(s):
                return {}
    
    # Build a single anchored regex enforcing order and exact-once,
    # allowing only spaces/newlines between blocks.
    parts = []
    for i, field in enumerate(hierarchy):
        tag = re.escape(field)
        # Capture inner content EXACTLY as-is (no surrounding \s* in the group)
        block = rf"<{tag}>(?P<g{i}>.*?)</{tag}>"
        parts.append(block)

    # Only spaces/newlines allowed between blocks; anchor ^...$
    between = r"[ \n]*"
    pattern = r"^" + between.join(parts) + r"$"
    rx = re.compile(pattern, re.DOTALL | re.IGNORECASE)

    m = rx.match(s)
    if not m:
        return {}

    # Build result preserving inner content verbatim
    result: Dict[str, str] = {}
    for i, field in enumerate(hierarchy):
        content = m.group(f"g{i}")
        # No strip here — preserve leading/trailing whitespace inside the tag
        result[field.strip()] = "" if content is None else content

    return result

@register("usim")
class UsimRewardManager(AbstractRewardManager):
    # Class-level shared dictionary to track global best hierarchies across all instances
    hierarchy_map = {}
    _update_hierarchy_lock = threading.Lock()

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        train_metrics: dict,
        val_metrics: dict = None,
        enable_hierarchy: dict = False,
        separate_generation: bool = False,
        strict_format: bool = False,
        fetch_global_best_hierarchy: bool = False,
        reward_fn_key: str = "data_source",
        compute_score: Optional[Callable] = None,
        hierarchy_config: str = None,
        states_in_think: bool = False,
        eval_push_to_hub: str = None,
        additional_generation_prompt: str = "",
        debug_logs: bool = False,
        debug_logs_max_examples_per_batch: int = 1,
        debug_logs_print_text: bool = False,
        debug_logs_text_max_chars: int = 1200,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score or default_compute_score
        self.split = 'train' if self.num_examine == 0 else 'val'
        
        self.strict_format = strict_format
        self.field_to_metrics = train_metrics if self.split == 'train' else val_metrics
        self.field_to_metrics = OmegaConf.to_container(self.field_to_metrics, resolve=True)
        self.metric_common_kwargs = self.field_to_metrics.pop('common_kwargs', {})

        self.separate_generation = separate_generation
        self.reward_fn_key = reward_fn_key
        self.enable_hierarchy = enable_hierarchy
        self.fetch_global_best_hierarchy = fetch_global_best_hierarchy
        self.add_additional_generation_prompt = not additional_generation_prompt == ''
        
        # Debug logging (opt-in)
        self.debug_logs = as_bool(debug_logs) or as_bool(os.getenv("USIM_DEBUG_LOGS", "0"))
        try:
            self.debug_logs_max_examples_per_batch = int(debug_logs_max_examples_per_batch)
        except Exception:
            self.debug_logs_max_examples_per_batch = 1
        self.debug_logs_print_text = as_bool(debug_logs_print_text)
        try:
            self.debug_logs_text_max_chars = int(debug_logs_text_max_chars)
        except Exception:
            self.debug_logs_text_max_chars = 1200

        # assert self.enable_hierarchy if self.separate_generation else True, \
        #     "enable_hierarchy must be True if separate_generation is True"
        self.states_in_think = states_in_think
        if self.enable_hierarchy:
            assert hierarchy_config, "hierarchy_config must be provided if enable_hierarchy is True"
            self.hierarchy_config = json.loads(open(hierarchy_config, 'r').read())
            self.hierarchy = list(self.hierarchy_config.keys())
            for h in self.hierarchy:
                if h not in self.field_to_metrics and h != 'think':
                    self.field_to_metrics[h] = {"state_reward": {"weight": 1.0, "kwargs": {}}}
                    print(f"Using default hierarchy_reward metric for hierarchy field {h}")
                self.hierarchy_map.setdefault(h, {})
        elif hierarchy_config:
            self.hierarchy_config = json.loads(open(hierarchy_config, 'r').read())
            self.hierarchy = list(self.hierarchy_config.keys())
            # for train set, we use default metric for each hierarchy (state_reward)
            if self.split == 'train':
                for h in self.hierarchy:
                    if h not in self.field_to_metrics:
                        self.field_to_metrics[h] = {"state_reward": {"weight": 1.0, "kwargs": {}}}
                        print(f"Using default hierarchy_reward metric for hierarchy field {h}")
        else:
            self.hierarchy = []

        if self.metric_common_kwargs:
            for field in self.field_to_metrics:
                for metric in self.field_to_metrics[field]:
                    self.field_to_metrics[field][metric]['kwargs'] = {
                        **self.metric_common_kwargs,
                        **self.field_to_metrics[field][metric].get('kwargs', {})
                    }
        
        self.field_metric_weights = {
            f"{field}:{metric}": weight_n_kwargs['weight'] \
            for field in self.field_to_metrics \
            for metric, weight_n_kwargs in self.field_to_metrics[field].items()
        }
        self.separate_rewards = kwargs.get("separate_rewards", False)
        
        # in pangram accumulator since val batch size is 1 we must accumulate the first
        # 100 generations indexed by the dataset index (extra_info['index'])
        print(self.field_to_metrics)
        for field, metrics in self.field_to_metrics.items():
            if 'pangram_score' in metrics:
                self.pangram_config = metrics.pop("pangram_score", None)
                self.pangram_kwargs = self.pangram_config['kwargs']
                self.pangram_enabled = True
            else:
                self.pangram_enabled = False
            
            if self.pangram_enabled:
                self.reset_pangram_accumulator()
                print(f"[Pangram] Enabled with config: {self.pangram_config}")

        print(f'split {self.split} | field_to_metrics {self.field_to_metrics}')

        self.eval_llm_api = kwargs.get("eval_llm_api", False)
        self.enable_thinking = kwargs.get("enable_thinking", False)
        # Eval only settings
        self.eval_push_to_hub = eval_push_to_hub
        self.eval_cache = {}
    
    def reset_pangram_accumulator(self):
        # reset accumulator and should be called at start of val
        # each gen in generations is of the form: (index, text) 
        self._pangram_accumulator = {
            "generations": [],  
            "evaluated": False,
            "results": None,
        }
    
    def accumulate_for_pangram(self, index: int, generation: str) -> bool:
       # if pangram accumulator
        if self._pangram_accumulator is None or index >= self.pangram_kwargs['max_samples']:
            return False
        if any(idx == index for idx, _ in self._pangram_accumulator["generations"]):
            return False
        self._pangram_accumulator["generations"].append((index, generation))
        return True
    
    async def finalize_pangram(self) -> dict:
        # run pangram on the accumulated samples
        if self._pangram_accumulator is None or self._pangram_accumulator["evaluated"]:
            return self._pangram_accumulator.get("metrics", {}) if self._pangram_accumulator else {}
        
        samples = sorted(self._pangram_accumulator["generations"], key=lambda x: x[0])
        if not samples:
            return {}
        
        generations = [text for _, text in samples]
        print(f"[Pangram] Running evaluation on {len(generations)} samples...")
        
        try:
            metric_path = Path(__file__).parent / "metrics" / "pangram_score.py"
            with METRIC_IMPORT_LOCK:
                if "pangram_score" not in sys.modules:
                    spec = importlib.util.spec_from_file_location("pangram_score", metric_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["pangram_score"] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules["pangram_score"]

            raw_results = await module.compute_score(
                data_source=None,
                generation=generations,
                ground_truth=None,
                extra_info={},
                **self.pangram_kwargs,
            )
            metrics = {
                "val/pangram/avg_ai_likelihood": raw_results[0]["avg_ai_likelihood"], 
                "val/pangram/total_words": raw_results[0]["total_word_count"],
                "val/pangram/num_samples": len(generations),
                "val/pangram/max_ai_likelihood": raw_results[0]["max_ai_likelihood"], 
                "val/pangram/fraction_ai": raw_results[0]["fraction_ai"], 
                "val/pangram/fraction_mixed": raw_results[0]["fraction_mixed"], 
                "val/pangram/fraction_human": raw_results[0]["fraction_human"], 
            }
            
            self._pangram_accumulator["evaluated"] = True
            self._pangram_accumulator["results"] = raw_results
            self._pangram_accumulator["metrics"] = metrics
            
            return metrics, raw_results[0]['windows']
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Pangram] Error: {e}")
            return {"val/pangram/error": str(e)}, None
        
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # Use asyncio.run to handle the async computation
        return asyncio.run(self._compute_rewards_async(data, return_dict))
    
    def state_dict(self) -> dict:
        # For saving and loading global hierarchy map
        with self._update_hierarchy_lock:
            return copy.deepcopy(self.hierarchy_map)
    
    def load_state_dict(self, state: dict) -> None:
        # resume from given hierarchy map
        if not isinstance(state, dict):
            raise ValueError("[ERROR] saved hierarchy map is not a dict")
        with self._update_hierarchy_lock:
            self.hierarchy_map = copy.deepcopy(state)

    def _stable_prompt_key(self, extra_info: dict | None) -> str:
        assert 'index' in extra_info, "extra_info must contain 'index' for stable key"
        return f"{extra_info['hierarchy_name']}:{extra_info['index']}"

    def compute_batch_score_sync(self, *args, **kwargs):
        result = asyncio.run(self.compute_score(*args, **kwargs))
        return result
    
    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False, eval_llm_api=None) -> torch.Tensor | dict[str, Any]:
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        
        data_source = data.non_tensor_batch["data_source"]
        batch_size = len(data_source)

        extra_info = data.non_tensor_batch["extra_info"]
        if len(self.hierarchy_config) == 1:
            extra_info = [
                {**info, "hierarchy_name": "response", "hierarchy_desc": self.hierarchy_config["response"]["desc"]}
                for info in extra_info
            ]
        else:
            extra_info = [
                {**info, "hierarchy_desc": self.hierarchy_config[info["hierarchy_name"]]["desc"]}
                for info in extra_info
            ]
        ground_truth = [item["ground_truth"] for item in data.non_tensor_batch["reward_model"]]
        keys = [self._stable_prompt_key(info) for info in extra_info]

        hierarchy_names = [item["hierarchy_name"] for item in extra_info]
        
        #########################################################
        ###################### Get prompts ######################
        attention_mask = data.batch["attention_mask"][:, :prompt_ids.shape[-1]]
        prompt_ids_no_pad = [
            ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, attention_mask)
        ]
        prompts = self.tokenizer.batch_decode(prompt_ids_no_pad, skip_special_tokens=False)

        #########################################################
        ################ Parse generation fields ################
        raw_generations = self.tokenizer.batch_decode(
            data.batch["responses"],
            skip_special_tokens=True,
        )

        # add the additional generate tokens
        field_to_metrics_lst = []
        generations, generation_fields = [], []
        valid_mask, valid_mask_by_hierarchy = [], {}
        should_have_field_masks = {h: [] for h in self.hierarchy}

        index = extra_info[0].get("index", -1) if extra_info else -1
        for i, (g, h) in enumerate(zip(raw_generations, hierarchy_names)):
            if self.add_additional_generation_prompt:
                additional_generation_prompt = f"<{h}>"
            else:
                additional_generation_prompt = ""
            generations.append(additional_generation_prompt + g)
            
            if (self.separate_generation or self.separate_rewards) and self.split == 'train':
                parse_hierarchy = [h]
            else:
                cur_hierarchy_idx = self.hierarchy.index(h)
                parse_hierarchy = self.hierarchy[cur_hierarchy_idx:]
            
            # remove 'think' from parse_hierarchy if exists
            if 'think' in parse_hierarchy:
                parse_hierarchy.remove('think')

            if self.strict_format:
                if self.enable_thinking:
                    if not ('response' in parse_hierarchy) and self.states_in_think:
                        generation_field = parse_fields_strict_thinking('<think>' + g, ['think'])
                        if len(generation_field) == 1 and len(parse_hierarchy) == 1:
                            generation_field = {parse_hierarchy[0]: generation_field['think']}
                    else:
                        generation_field = parse_fields_strict_thinking(g, parse_hierarchy)
                else:
                    generation_field = parse_fields_strict(g, parse_hierarchy)
            else:
                generation_field = parse_any_fields(g)
            
            # Debug parsing detail (limit to a few examples per batch)
            if self.debug_logs and i < max(0, self.debug_logs_max_examples_per_batch):
                parsed_lens = {k: len(v) for k, v in generation_field.items()}
                expected_fields = list(parse_hierarchy)
                found_any_tags = list(parse_any_fields(g).keys())
                dbg_key = self._stable_prompt_key(extra_info[i])
                debug_print_kh(
                    True,
                    dbg_key,
                    h,
                    "parse: "
                    f"strict={self.strict_format} parse_hierarchy={parse_hierarchy} "
                    f"raw_gen_prefix='{shorten(g, 220)}' "
                    f"parsed_keys={list(generation_field.keys())} "
                    f"parsed_lens={parsed_lens}",
                )
                if self.strict_format:
                    missing = [f for f in expected_fields if f not in generation_field]
                    if missing:
                        debug_print_kh(
                            True,
                            dbg_key,
                            h,
                            "[PARSE_FAIL] "
                            f"missing={missing} expected={expected_fields} found_tags={found_any_tags} "
                            f"raw_gen_prefix='{shorten(g, 400)}'",
                        )
                if self.debug_logs_print_text:
                    gt_i = ground_truth[i] if i < len(ground_truth) else ""
                    debug_print_kh(
                        True,
                        dbg_key,
                        h,
                        f"TEXT raw_generation='{shorten(g, self.debug_logs_text_max_chars)}'",
                    )
                    debug_print_kh(
                        True,
                        dbg_key,
                        h,
                        f"TEXT ground_truth='{shorten(gt_i, self.debug_logs_text_max_chars)}'",
                    )
            
            if self.split == 'val':
                if self.debug_logs:
                    print(f"Filling in raw generation for hierarchy {h}: {g}", flush=True)
                if generation_field.get(h, "").strip() == '':
                    generation_field[h] = g.split('</think>')[-1].strip()
                    if (clean_field := generation_field[h].split(f'<{h}>')[-1].strip()) != '':
                        generation_field[h] = clean_field

            if self.separate_generation:
                field_to_metrics = {h: self.field_to_metrics.get(h)}
            else:
                # field_to_metrics = self.field_to_metrics BUG
                field_to_metrics = {h: self.field_to_metrics[h] for h in parse_hierarchy}

            for _h in self.hierarchy:
                if self.separate_generation:
                    should_have_field_masks[_h].append(h == _h)
                else:
                    should_have_field_masks[_h].append(True if _h in parse_hierarchy else False)

            field_to_metrics_lst.append(field_to_metrics)
            field_key = list(set(generation_field.keys()).intersection(set(field_to_metrics.keys())))
            
            is_valid = len(field_key) == len(field_to_metrics)
            valid_mask.append(float(is_valid))

            if self.strict_format:
                generation_fields.append(generation_field if is_valid else {})
            else:
                generation_fields.append(generation_field)
            
            # [PANGRAM]
            if self.split == 'val' and hasattr(self, '_pangram_accumulator') and self._pangram_accumulator is not None:
                idx = extra_info[i].get("index", i)
                gen_text = generation_field.get("response", g)
                self.accumulate_for_pangram(idx, gen_text)

            valid_mask_by_hierarchy.setdefault(h, [])
            valid_mask_by_hierarchy[h].append(float(is_valid))
            
        counts = Counter(hierarchy_names)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        should_have_field_masks = {field: torch.tensor(should_have_field_masks[field], dtype=torch.bool) for field in should_have_field_masks}
        # if self.debug_logs:
        print(
            f"\n| hierarchy_name {counts}\n| generation {generations[0]}\n| generation_fields {generation_fields[0]}\n"
            f"| ground_truth {ground_truth[0]} | all_valid_rate {valid_mask.float().mean().item()}\n| prompts {prompts[0]}",
            flush=True,
        )

        unique_keys = set(keys)
        generation_fields_by_keys = {
            key: [generation_fields[i] for i in range(batch_size) if keys[i] == key]
            for key in unique_keys
        }
        field_to_metrics_by_keys = {
            key: field_to_metrics_lst[i] for i, key in enumerate(keys)
        }

        # Map keys to their data and validate consistency
        key_to_data = {}
        for i, key in enumerate(keys):
            data_info = {
                'data_source': data_source[i],
                'ground_truth': ground_truth[i],
                'extra_info': extra_info[i]
            }
            if key in key_to_data:
                assert key_to_data[key] == data_info, \
                    "Data source, ground truth, and extra_info must be consistent for the same key"
            else:
                key_to_data[key] = data_info

        # Create async scoring tasks with deduplication
        tasks = []
        task_metadata = []  # Track (field, key, original_indices, unique_values) for each task
        
        #########################################################
        #################### Compute rewards ####################
        loop = asyncio.get_running_loop()
        for key in generation_fields_by_keys:
            field_to_metrics = field_to_metrics_by_keys[key]

            for field in field_to_metrics:
                metrics_to_run = {
                    m: v for m, v in field_to_metrics[field].items() 
                }
                if not metrics_to_run:
                    continue
                field_values = [gen_fields.get(field, '').strip() for gen_fields in generation_fields_by_keys[key]]
                
                # Deduplicate while preserving mapping
                unique_values = []
                value_to_unique_idx = {}
                original_to_unique_indices = []
                
                for val in field_values:
                    if val not in value_to_unique_idx:
                        value_to_unique_idx[val] = len(unique_values)
                        unique_values.append(val)
                    original_to_unique_indices.append(value_to_unique_idx[val])
                
                key_extra_info = copy.deepcopy(key_to_data[key]['extra_info'])
                key_extra_info.update({"hierarchy": self.hierarchy})
                key_extra_info.update({"hierarchy_name": field})
                key_extra_info.update({"hierarchy_desc": self.hierarchy_config[field]['desc']})
                if self.debug_logs:
                    key_extra_info["_usim_debug_logs"] = True
                    key_extra_info["_usim_key"] = key
                    if self.debug_logs_max_examples_per_batch > 0:
                        # Provide a short preview of what will be scored
                        debug_print_kh(
                            True,
                            key,
                            field,
                            "task: "
                            f"metrics={list(metrics_to_run.keys())} "
                            f"num_values={len(field_values)} num_unique_values={len(unique_values)} "
                            f"unique0='{shorten((unique_values[0] if unique_values else ''), 220)}'",
                        )
                        if self.debug_logs_print_text:
                            debug_print_kh(
                                True,
                                key,
                                field,
                                f"TEXT unique0_full='{shorten((unique_values[0] if unique_values else ''), self.debug_logs_text_max_chars)}'",
                            )
                            debug_print_kh(
                                True,
                                key,
                                field,
                                f"TEXT ground_truth='{shorten(key_to_data[key]['ground_truth'], self.debug_logs_text_max_chars)}'",
                            )
                
                task = loop.run_in_executor(
                    None,
                    partial(
                            self.compute_batch_score_sync,
                            key_to_data[key]["data_source"],
                            unique_values,
                            key_to_data[key]["ground_truth"],
                            field_to_metrics[field],
                            key_extra_info,
                            eval_llm_api=self.eval_llm_api,   # <- keyword passed through
                        ),
                    )
                tasks.append(task)
                task_metadata.append((field, key, original_to_unique_indices))

        # Execute tasks and reorganize scores
        gathered_scores = await asyncio.gather(*tasks)

        reward_dicts = [{} for _ in range(batch_size)]
        for task_idx, (field, key, original_to_unique_indices) in enumerate(task_metadata):
            # gathered_scores[task_idx] is a dict: {metric_name: [score1, score2, ...]}
            # where the list has one score per unique value
            metrics_dict = gathered_scores[task_idx]
            
            # Map scores back to original order for this key
            key_indices = [i for i in range(batch_size) if keys[i] == key]
            
            for local_idx, global_idx in enumerate(key_indices):
                unique_idx = original_to_unique_indices[local_idx]
                
                # Initialize field dict if not exists
                if field not in reward_dicts[global_idx]:
                    reward_dicts[global_idx][field] = {}
                
                # Store scores for each metric under this field
                for metric_name, scores_list in metrics_dict.items():
                    reward_dicts[global_idx][field][metric_name] = scores_list[unique_idx]
        
        # Update eval cache 
        if self.split == "val" and self.eval_push_to_hub:
            for i, (key, prompt, gen, raw_gen, gt, extra) in enumerate(
                zip(keys, prompts, generation_fields, raw_generations, ground_truth, extra_info)
            ):
                reward_dict = reward_dicts[i]
                self.eval_cache[key] = {
                    "prompt": prompt,
                    "raw_generation": raw_gen,
                    "response": gen.get('response', ''),
                    "ground_truth": gt,
                    "metrics": json.dumps({
                        field: {k: v for k, v in m.items() if not k.endswith("metrics_info")}
                        for field, m in reward_dict.items()
                    }),
                    "metrics_info": json.dumps({
                        field: {k: v for k, v in m.items() if k.endswith("metrics_info")}
                        for field, m in reward_dict.items()
                    }),
                    "extra_info": json.dumps(extra),
                }
        
        # Now reward_dicts[i][field][metric] gives the score for batch_i, field, metric
        # Aggregate scores for each field:metric combination
        if self.separate_generation or self.separate_rewards:
            # Aggregate scores - each sample only has scores for its own hierarchy
            scores_by_fm = {}
            scores = torch.zeros(batch_size)

            for i in range(batch_size):
                h = hierarchy_names[i]  # This sample's target hierarchy
                reward_dict = reward_dicts[i]
                
                # No valid parse
                if h not in reward_dict:
                    import pdb; pdb.set_trace()
                
                sample_score = 0.0
                for metric_key, value in reward_dict[h].items():
                    if metric_key.endswith('metrics_info'):
                        continue
                    
                    fm_key = f"{h}:{metric_key}"
                    # state_reward from state_reward:score
                    base_metric = metric_key.split(':')[0]  
                    weight_key = f"{h}:{base_metric}"
                    weight = self.field_metric_weights.get(weight_key, 1.0)
                    
                    weighted_value = value * weight
                    sample_score += weighted_value
                    
                    # Track for logging
                    scores_by_fm.setdefault(fm_key, torch.zeros(batch_size))
                    scores_by_fm[fm_key][i] = weighted_value
                
                scores[i] = sample_score
            weighted_scores_by_fm = scores_by_fm
        else:
            scores_by_fm = {}
            try:
                #iterate_fields = self.field_to_metrics
                # TODO: we assume separate_rewards doesn't have mixed batches
                # if self.separate_generation or (self.separate_rewards and self.split == "train"):
                #     iterate_fields = set(hierarchy_names)
                for field in self.field_to_metrics:
                    for metric in self.field_to_metrics[field]:
                        sub_metrics = set()
                        for reward_dict in reward_dicts:
                            field_data = reward_dict.get(field, {})
                            for key in field_data.keys():
                                if key.startswith(metric) and not key.endswith('metrics_info'):
                                    sub_metrics.add(key)

                        for sub_metric in sorted(sub_metrics):
                            fm_key = f"{field}:{sub_metric}"
                            # ERROR field may not be in reward_dicts
                            scores_by_fm[fm_key] = torch.tensor([
                                reward_dicts[i][field][sub_metric]
                                for i in range(batch_size)
                            ])
                        self.field_metric_weights.update({
                            f"{field}:{sub_metric}": self.field_metric_weights[f"{field}:{metric}"]
                            for sub_metric in sub_metrics
                        })
            except: 
                import pdb; pdb.set_trace()

            # Apply field-metric specific weights
            weighted_scores_by_fm = {
                fm: scores_by_fm[fm] * self.field_metric_weights[fm]
                for fm in scores_by_fm
            }

        # Combine weighted scores from all field and metrics into a single tensor
        if scores_by_fm:
            scores = torch.stack(
                [weighted_scores_by_fm[fm] for fm in scores_by_fm]
            ).sum(dim=0)
        else:
            scores = torch.zeros(batch_size)

        # Compute mean of weighted scores for each metric
        if scores_by_fm:
            log_weighted_scores_by_field_metric = {
                f"{self.split}/{fm}": weighted_scores_by_fm[fm][should_have_field_masks[fm.split(":")[0]]].mean(dim=0).item()
                for fm in scores_by_fm
            }
            # log_weighted_scores_by_field_metric.update({
            #     f"{self.split}/valid:{fm}": (
            #         weighted_scores_by_fm[fm][valid_mask.bool()].mean(dim=0).item()
            #         if valid_mask.any() else 0.0
            #     )
            #     for fm in scores_by_fm
            # })
        else:
            log_weighted_scores_by_field_metric = {}

        # constrain the minimum score to be a very small number, so we can use it
        # to track the hiererachical generation parts in the grpo_prm implementation
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
        
        #########################################################
        ############ Update global best hierarchy map ############
        if self.enable_hierarchy and self.split == 'train':
            with self._update_hierarchy_lock:

                for i in range(batch_size):
                    key = keys[i]
                    hierarchy_name = hierarchy_names[i]

                    next_hierarchy_idx = self.hierarchy.index(hierarchy_name)
                    remaining_hierarchy_idx = range(next_hierarchy_idx, len(self.hierarchy) - 1)

                    if self.separate_rewards or (self.separate_generation and remaining_hierarchy_idx):
                        if len(remaining_hierarchy_idx) > 0:
                            remaining_hierarchy_idx = [remaining_hierarchy_idx[0]]
                    
                    data_i = copy.deepcopy(data[i].non_tensor_batch)
                    data_i["prompt"] = prompts[i]
                    
                    for hierarchy_idx in remaining_hierarchy_idx:
                        h = self.hierarchy[hierarchy_idx]
                        next_h = self.hierarchy[hierarchy_idx + 1]
                        
                        # get current score of hierarchy name
                        metric = list(self.field_to_metrics[h].keys())[0]
                        current_score = scores_by_fm[f"{h}:{metric}:score"][i].item()

                        field = generation_fields[i].get(h, "")
                        
                        if len(field.strip(" \n")) == 0 or \
                            current_score < self.hierarchy_config[h]['threshold']:
                            break

                        if self.add_additional_generation_prompt:
                            append_to_prompt = HALF_HIERARCHY_TEMPLATE.format(name=h, field=field) + f"\n<{next_h}>\n" 
                        else:
                            append_to_prompt = HIERARCHY_TEMPLATE.format(name=h, field=field) + "\n" 
                        data_i["prompt"] = data_i["prompt"] + append_to_prompt
                        
                        data_i["extra_info"]["hierarchy_name"] = next_h
                        current_entry = {
                            "reward": current_score,
                            "field": field,
                            "data": data_i,
                            "timestamp": time.time() # put the latest one first if tie in reward
                        }

                        if self.fetch_global_best_hierarchy:
                            self.hierarchy_map[h].setdefault(key, [])
                            entries = self.hierarchy_map[h][key]
                            entries.append(current_entry)
                            
                            unique_map = {}
                            for d in sorted(
                                entries, reverse=True,
                                key=lambda x: (x["reward"], x["timestamp"])
                            ):
                                if d["field"] not in unique_map:
                                    unique_map[d["field"]] = d
                            unique_entries = list(unique_map.values())
                        else:
                            # Only store the most recent entry
                            unique_entries = [current_entry]

                        # Sort primarily by reward (desc), tie-break by timestamp (desc)
                        self.hierarchy_map[h][key] = sorted(
                            unique_entries,
                            key=lambda x: (x["reward"], x["timestamp"]),
                            reverse=True,
                        )[:1]

                        data_i = copy.deepcopy(data_i)

            log_dataset_size = {
                f"{self.split}/hmap_size/{hname}": len(self.hierarchy_map[hname]) for hname in self.hierarchy_map
            }
            log_weighted_scores_by_field_metric.update(log_dataset_size)
        
        #########################################################
        log_weighted_scores_by_field_metric.update(
            {
                **{
                    f"{self.split}/valid_rate/{h if self.separate_generation else 'all'}": torch.tensor(valid_rate).mean().item() for h, valid_rate in valid_mask_by_hierarchy.items()
                },
                **{
                    f"{self.split}/sample_size_in_batch/{h}": counts[h] for h in valid_mask_by_hierarchy.keys()
                }

            }
        )
        if self.debug_logs:
            print('log_weighted_scores_by_field_metric', log_weighted_scores_by_field_metric, flush=True)
        if return_dict:
            return {
                "reward_tensor": reward_tensor, 
                "reward_extra_info": {**log_weighted_scores_by_field_metric},
                "reward_dicts": reward_dicts
            }
        else:
            return reward_tensor, {**log_weighted_scores_by_field_metric}