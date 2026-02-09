
import asyncio
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
import copy
import psutil
import torch
import threading, time
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
import os
import re
import json
import torch.distributed as dist
from omegaconf import OmegaConf
from typing import Dict, Any, Tuple, Optional, List, TypedDict

HIERARCHY_TEMPLATE = "<{name}>{field}</{name}>"
FIELD_PATTERN = re.compile(r"<(?P<name>\w+)>(?P<field>.*?)</\1>", re.DOTALL | re.IGNORECASE)

def parse_fields(text: str) -> Dict[str, str]:
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
        additional_generation_prompt: str = "",
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score or default_compute_score
        self.split = 'train' if self.num_examine == 0 else 'val'
        
        self.strict_format = strict_format
        self.field_to_metrics = train_metrics if self.split == 'train' else val_metrics
        self.field_to_metrics = OmegaConf.to_container(self.field_to_metrics, resolve=True)

        self.separate_generation = separate_generation
        self.reward_fn_key = reward_fn_key
        self.enable_hierarchy = enable_hierarchy
        self.fetch_global_best_hierarchy = fetch_global_best_hierarchy
        self.additional_generation_prompt = additional_generation_prompt

        assert self.enable_hierarchy if self.separate_generation else True, \
            "enable_hierarchy must be True if separate_generation is True"

        if self.enable_hierarchy:
            assert hierarchy_config, "hierarchy_config must be provided if enable_hierarchy is True"
            self.hierarchy_config = json.loads(open(hierarchy_config, 'r').read())
            self.hierarchy = list(self.hierarchy_config.keys())
            for h in self.hierarchy:
                if h not in self.field_to_metrics:
                    self.field_to_metrics[h] = {"hierarchy_reward": {"weight": 1.0, "kwargs": {}}}
                    print(f"Using default hierarchy_reward metric for hierarchy field {h}")
                self.hierarchy_map.setdefault(h, {})

        self.field_metric_weights = {
            f"{field}:{metric}": weight_n_kwargs['weight'] \
            for field in self.field_to_metrics \
            for metric, weight_n_kwargs in self.field_to_metrics[field].items()
        }
        

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # Use asyncio.run to handle the async computation
        return asyncio.run(self._compute_rewards_async(data, return_dict))
    
    def state_dict(self) -> dict:
        # For saving and loading global hierarchy map
        with self._update_hierarchy_lock:
            return {"hierarchy_map": copy.deepcopy(self.hierarchy_map)}
    
    def load_state_dict(self, state: dict) -> None:
        # resume from given hierarchy map
        if not isinstance(state, dict):
            raise ValueError("[ERROR] saved hierarchy map is not a dict")
        hm = state.get("hierarchy_map", {}) 
        with self._update_hierarchy_lock:
            self.hierarchy_map = copy.deepcopy(hm)

    def _stable_prompt_key(self, extra_info: dict | None) -> str:
        assert 'index' in extra_info, "extra_info must contain 'index' for stable key"
        return f"{extra_info['hierarchy_name']}:{extra_info['index']}"

    def compute_score_sync(self, *args, **kwargs):
        result = asyncio.run(self.compute_score(*args, **kwargs))
        return result

    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        
        data_source = data.non_tensor_batch["data_source"]
        batch_size = len(data_source)

        extra_info = data.non_tensor_batch["extra_info"]
        extra_info = [
            {**info, "hierarchy_desc": self.hierarchy_config[info["hierarchy_name"]]["desc"]}
            for info in extra_info
        ]

        ground_truth = [item["ground_truth"] for item in data.non_tensor_batch["reward_model"]]
        keys = [self._stable_prompt_key(info) for info in extra_info]
        hierarchy_names = [item["hierarchy_name"] for item in extra_info]
        hierarchy_name = hierarchy_names[0]

        assert len(set(hierarchy_names)) == 1, "All items in the batch must have the same hierarchy_name"
        assert hierarchy_name in self.field_to_metrics, f"Hierarchy name {hierarchy_name} not in field_to_metrics"

        #########################################################
        ################ Parse generation fields ################
        generations = self.tokenizer.batch_decode(
            data.batch["responses"],
            skip_special_tokens=True,
        )
        # add the additional generate tokens
        generations = [self.additional_generation_prompt + gen for gen in generations]
        if self.strict_format:
            if  self.separate_generation and self.split == 'train':
                parse_hierarchy = [hierarchy_name]
            else:
                parse_hierarchy = self.hierarchy

            generation_fields = [parse_fields_strict(generation, parse_hierarchy) for generation in generations]
        else:
            generation_fields = [parse_fields(generation) for generation in generations]

        #########################################################
        ###################### Get prompts ######################
        attention_mask = data.batch["attention_mask"][:, :prompt_ids.shape[-1]]
        prompt_ids_no_pad = [
            ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, attention_mask)
        ]
        prompts = self.tokenizer.batch_decode(prompt_ids_no_pad, skip_special_tokens=False)

        #########################################################
        #################### Compute rewards ####################
        if self.separate_generation and self.split == 'train':
            field_to_metrics = {hierarchy_name: self.field_to_metrics.get(hierarchy_name)}
        else:
            field_to_metrics = self.field_to_metrics
        
        field_keys = [list(set(g.keys()).intersection(set(field_to_metrics.keys()))) for g in generation_fields]
        valid_mask = torch.tensor([1 if len(k) == len(field_to_metrics) else 0 for k in field_keys])
        valid_rate = valid_mask.float().mean().item()

        print(f"\n| hierarchy_name {hierarchy_name}\n| generation {generations[0]}\n| generation_fields {generation_fields[0]}\n| ground_truth {ground_truth[0]} | valid_rate {valid_rate}\n| prompts {prompts[0][-20:]}")
        loop = asyncio.get_running_loop()

        if self.strict_format:
            generation_fields = [
                field if is_valid else {}
                for is_valid, field in zip(valid_mask, generation_fields)
            ]

        tasks = [
            loop.run_in_executor(          
                    None,                      
                    self.compute_score_sync,
                    data_source[i], generation_fields[i].get(field, None), ground_truth[i],
                    field_to_metrics[field], extra_info[i],
                )
                for field in field_to_metrics for i in range(batch_size) 
        ]
        score_dicts = await asyncio.gather(*tasks)
        field_to_score_dict = {
            field: score_dicts[i * batch_size:(i + 1) * batch_size]
            for i, field in enumerate(field_to_metrics)
        }

        # Aggregate scores for each metric and field 
        scores_by_fm = {
            f"{field}:{metric}": torch.tensor(
                [score_dict[metric] for score_dict in field_to_score_dict[field]]
            ) for field in field_to_metrics for metric in field_to_metrics[field]
        }

        # Apply field-metric specific weights
        weighted_scores_by_fm = {
            fm: scores_by_fm[fm] * self.field_metric_weights[fm]
            for fm in scores_by_fm
        }

        # Compute mean of weighted scores for each metric
        log_weighted_scores_by_field_metric = {
            f"{self.split}/{fm}": weighted_scores_by_fm[fm].mean(dim=0).item()
            for fm in scores_by_fm
        }
        log_weighted_scores_by_field_metric.update({
            f"{self.split}/valid:{fm}": (
                weighted_scores_by_fm[fm][valid_mask.bool()].mean(dim=0).item()
                if valid_mask.any() else 0.0
            )
            for fm in scores_by_fm
        })

        # Combine weighted scores from all field and metrics into a single tensor
        scores = torch.stack(
            [weighted_scores_by_fm[fm] for fm in scores_by_fm]
        ).sum(dim=0)

        # constrain the minimum score to be a very small number, so we can use it
        # to track the hiererachical generation parts in the grpo_prm implementation
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        #########################################################
        ############ Update global best hierarchy map ############
        if self.enable_hierarchy and self.split == 'train':
            with self._update_hierarchy_lock:

                import pdb; pdb.set_trace()
                for i in range(batch_size):
                    key = keys[i]

                    next_hierarchy_idx = self.hierarchy.index(hierarchy_name)
                    remaining_hierarchy_idx = range(next_hierarchy_idx, len(self.hierarchy) - 1)

                    if self.separate_generation and remaining_hierarchy_idx:
                        remaining_hierarchy_idx = [remaining_hierarchy_idx[0]]
                    
                    data_i = copy.deepcopy(data[i].non_tensor_batch)
                    data_i["prompt"] = prompts[i]
                    
                    import pdb; pdb.set_trace()
                    for hierarchy_idx in remaining_hierarchy_idx:
                        h = self.hierarchy[hierarchy_idx]
                        next_h = self.hierarchy[hierarchy_idx + 1]

                        # get current score of hierarchy name
                        metric = list(self.field_to_metrics[hierarchy_name].keys())[0]
                        current_score = scores_by_fm[f"{hierarchy_name}:{metric}"][i]
                        field = generation_fields[i].get(hierarchy_name, "")

                        if len(field.strip(" \n")) == 0 or \
                            current_score < self.hierarchy_config[hierarchy_name]['threshold']:
                            break
                        
                        data_i["prompt"] = data_i["prompt"] + HIERARCHY_TEMPLATE.format(name=hierarchy_name, field=field) + "\n" 
                        data_i["extra_info"]["hierarchy_name"] = next_h
                        
                        current_entry = {
                            "reward": current_score,
                            "field": field,
                            "data": data_i,
                            "timestamp": time.time() # put the latest one first if tie in reward
                        }

                        if self.fetch_global_best_hierarchy:
                            entries = self.hierarchy_map[hierarchy_name].setdefault(key, [])
                            entries.append(current_entry)
                            
                            unique_map = {}
                            for d in sorted(
                                entries, 
                                key=lambda x: (x["reward"], x["timestamp"]), 
                                reverse=True
                            ):
                                if d["field"] not in unique_map:
                                    unique_map[d["field"]] = d
                            unique_entries = list(unique_map.values())
                        else:
                            # Only store the most recent entry
                            unique_entries = [current_entry]

                        import pdb; pdb.set_trace()
                        # Sort primarily by reward (desc), tie-break by timestamp (desc)
                        self.hierarchy_map[hierarchy_name][key] = sorted(
                            unique_entries,
                            key=lambda x: (x["reward"], x["timestamp"]),
                            reverse=True,
                        )[:1]

                        data_i = copy.deepcopy(data_i)

            log_dataset_size = {
                f"{self.split}/{hname}/size": len(self.hierarchy_map[hname]) for hname in self.hierarchy_map
            }
            log_weighted_scores_by_field_metric.update(log_dataset_size)

        #########################################################
        log_weighted_scores_by_field_metric.update(
            {
                f"{self.split}/{hierarchy_name if self.separate_generation else 'all'}/valid_rate": valid_rate,
            }
        )
        print('log_weighted_scores_by_field_metric', log_weighted_scores_by_field_metric)
        if return_dict:
            return {
                "reward_tensor": reward_tensor, 
                "reward_extra_info": {**log_weighted_scores_by_field_metric}
            }
        else:
            return reward_tensor, {**log_weighted_scores_by_field_metric}