# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from recipe.usim.create_any_dataset import format_persona

logger = logging.getLogger(__name__)


SYS_START = "<|im_start|>system\n"
SYS_END   = "<|im_end|>\n"


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        is_train: bool = True,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.additional_generation_prompt = self.config.get("additional_generation_prompt", '')

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        multirole_chat_template_path = config.kwargs.get("multirole_chat_template_path")

        # [USIM] if there is a chat_template path, we have to pass in speak_as when add_generation_prompt is true
        self.speak_as = True if multirole_chat_template_path else False

        # ======= [USIM] system prompts for each hierarchy
        self.hierarchy_config_path = config.get("hierarchy_config_path", None)
        self.hierarchy_system_prompts = {}

        self.augment_with_hierarchies = config.get("augment_with_hierarchies", False)
        self.enable_hetero_think = config.get("enable_hetero_think", False)
        self.is_train = is_train
        self.field_dropout_prob = config.get("field_dropout_prob", 0.0) if is_train else 0.0
        self.item_dropout_prob = config.get("item_dropout_prob", 0.0) if is_train else 0.0
        self.hierarchy_names = [] 

        self.separate_rewards = config.get("separate_rewards", False)
        self.separate_generation = config.get("separate_generation", False)
        self.val_size = config.get("val_size", 2000)
        self.dataset = config.get("dataset", None)
        self.eval_hierarchy_name = config.get("eval_hierarchy_name", "response")
        self.eval_only=config.get("eval_only", False)

        self.is_wildchat=config.get("is_wildchat", False)

        self.generate_hierarchies=config.get("generate_hierarchies", False)



        self.new_sys_prompt = config.get("new_sys_prompt", None)
        if len(self.new_sys_prompt) == 0:
            self.new_sys_prompt = None

        if self.new_sys_prompt is not None:
            self.new_sys_prompt_text = self._load_single_hierarchy_prompt(self.new_sys_prompt)
        
        assert not (self.augment_with_hierarchies and self.new_sys_prompt is not None)

        
        if self.hierarchy_config_path and self.augment_with_hierarchies:
            self._load_hierarchy_system_prompts()
        # ======= 

        self._download()
        self._read_files_and_tokenize()

        # ======= [USIM] 
        self._original_len = len(self.dataframe)
    
    def _load_hierarchy_system_prompts(self):
        """Load hierarchy config and read system prompt files into memory."""
        import json
        
        with open(self.hierarchy_config_path, 'r') as f:
            hierarchy_config = json.load(f)
        
        self.hierarchy_names = list(hierarchy_config.keys())
        
        for hierarchy_name, cfg in hierarchy_config.items():
            system_prompt_path = cfg.get('system_prompt')
            if system_prompt_path:
                if not os.path.isabs(system_prompt_path):
                    # Make relative paths relative to hierarchy config location
                    base_dir = os.path.dirname(self.hierarchy_config_path)
                    system_prompt_path = os.path.join(base_dir, system_prompt_path)
                
                if os.path.exists(system_prompt_path):
                    with open(system_prompt_path, 'r') as f:
                        self.hierarchy_system_prompts[hierarchy_name] = f.read().strip()
                    print(f"Loaded system prompt for '{hierarchy_name}' from {system_prompt_path}")
                else:
                    print(f"Warning: System prompt file not found: {system_prompt_path}")
        print(self.hierarchy_system_prompts)
        print(f"Loaded {len(self.hierarchy_system_prompts)} hierarchy system prompts")

    def _load_single_hierarchy_prompt(self, system_prompt_path):
        import json
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, 'r') as f:
                new_system_prompt = f.read().strip()

            print(f"Loaded NEW SYSTEM PROMPT FROM {system_prompt_path}")
        else:
            print(f"Warning: System prompt file not found: {system_prompt_path}")

        return new_system_prompt
    def _replace_system_prompt(self, messages: list, new_system_prompt: str) -> list:
        """Replace or inject a system prompt into the message list."""
        if not messages:
            return [{"role": "system", "content": new_system_prompt}]
        
        messages = copy.deepcopy(messages)
        if messages[0]["role"] == "system":
            messages[0]["content"] = new_system_prompt
        else:
            messages.insert(0, {"role": "system", "content": new_system_prompt})
        
        return messages

    def _persona_dropout(self, row_dict, messages: list) -> list:
        """Replace the persona with one based on dropout probability."""

        extra_info = row_dict.get('extra_info')
        assert 'persona' in extra_info, "persona_dropout is enabled but no persona found in extra_info"
        
        persona = extra_info['persona']
        if isinstance(persona, str):
            import json
            persona = json.loads(persona)
        assert isinstance(persona, dict), "persona in extra_info should be a dict"
        
        # replace the persona with a dropped one from the system prompt
        new_messages = copy.deepcopy(messages)
        if new_messages[0]["role"] == "system":
            system_content = new_messages[0]["content"]
            pattern = r"<\|The Start of Persona\|>.*?<\|The End of Persona\|>"

            dropout_persona = persona.copy()
            formatted_persona = format_persona(dropout_persona, self.field_dropout_prob, self.item_dropout_prob)
            
            formatted_persona_with_tags = f"<|The Start of Persona|>\n{formatted_persona}\n<|The End of Persona|>"
            
            new_system_content = re.sub(pattern, lambda m: formatted_persona_with_tags, system_content, flags=re.DOTALL)
            new_messages[0]["content"] = new_system_content
        return new_messages
        
    def _get_hierarchy_system_prompt(self, row_dict: dict, hierarchy_name: str) -> Optional[str]:
        """Get the formatted system prompt for a specific hierarchy level."""
        if not self.hierarchy_system_prompts:
            return None
        
        template = self.hierarchy_system_prompts.get(hierarchy_name)
        if template is None:
            return None

        extra_info = row_dict.get('extra_info')
        if extra_info is None:
            extra_info = {}
        elif not isinstance(extra_info, dict):
            extra_info = {}

        format_dict = dict(extra_info)
    
        if 'persona' in format_dict:
            if isinstance(format_dict['persona'], str):
                import json
                try:
                    format_dict['persona'] = json.loads(format_dict['persona'])
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(format_dict['persona'], dict):
                format_dict['persona'] = format_persona(format_dict['persona'])
        
        try:
            formatted_prompt = template.format(**format_dict)
            return formatted_prompt
        except KeyError as e:
            print(f"Warning: Missing key {e} for system prompt template")
            return None
        except Exception as e:
            print(f"Warning: Failed to format system prompt: {e}")
            return None
    
    def _get_new_system_prompt(self, row_dict: dict, template) -> Optional[str]:
        if template is None:
            return None

        extra_info = row_dict.get('extra_info')
        if extra_info is None:
            extra_info = {}

        elif not isinstance(extra_info, dict):
            extra_info = {}
        format_dict = dict(extra_info)

        if 'persona' in format_dict:
            if isinstance(format_dict['persona'], str):
                import json
                try:
                    format_dict['persona'] = json.loads(format_dict['persona'])
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(format_dict['persona'], dict):
                format_dict['persona'] = format_persona(format_dict['persona'])
        try:
            formatted_prompt = template.format(**format_dict)
            return formatted_prompt

        except KeyError as e:
            print(f"Warning: Missing key {e} for system prompt template")
            return None
        except Exception as e:
            print(f"Warning: Failed to format system prompt: {e}")
            return None

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        
        # # --- BEFORE ---
        # def extract_gt(rm):
        #     if isinstance(rm, dict):
        #         return rm.get("ground_truth")
        #     if isinstance(rm, str):
        #         try:
        #             obj = json.loads(rm)
        #             if isinstance(obj, dict):
        #                 return obj.get("ground_truth")
        #         except Exception:
        #             return None
        #     return None

        # def has_at(example):
        #     gt = extract_gt(example.get("reward_model"))
        #     return isinstance(gt, str) and gt.startswith("@")
        # before_len = len(self.dataframe)
        # before_at = len(self.dataframe.filter(has_at, num_proc=1, desc="count @ before"))
        # print(f"[before] total={before_len}  @rows={before_at}")

        # [USIM] filter out rows where ground truth starts w/ @
        if self.dataset == "youtube":
            def keep_row(batch):
                keep = []
                for rm in batch.get("reward_model", []):
                    gt = None
                    if isinstance(rm, dict):
                        gt = rm.get("ground_truth")
                    elif isinstance(rm, str):
                        try:
                            rm_obj = json.loads(rm)
                            if isinstance(rm_obj, dict):
                                gt = rm_obj.get("ground_truth")
                        except Exception:
                            raise ValueError("NO GT FOUND")
                    else:
                        raise ValueError("RM NOT STR OR DICT")
                    print(gt)
                    keep.append(not (isinstance(gt, str) and gt.startswith("@")))
                return keep

            self.dataframe = self.dataframe.filter(
                keep_row,
                batched=True,
                num_proc=self.num_workers,
                desc='Filtering reward_model.ground_truth starting with "@"',
            )

        #[USIM]
        if not self.is_train:
            if self.eval_only:
                self.dataframe = self.dataframe.select(range(min(self.val_size, len(self.dataframe))))
            else:
                print("[WARNING] SHUFFLING val set IF YOU SEE THIS MAKE SURE YOU'RE NOT RUNNING TESTING")
                self.dataframe = self.dataframe.shuffle(seed=42).select(range(min(self.val_size, len(self.dataframe))))

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

        #DEBUGGING
        # after_len = len(self.dataframe)
        # after_at = len(self.dataframe.filter(has_at, num_proc=1, desc="count @ after"))
        # print(f"[after] total={after_len}  @rows={after_at}  dropped={before_len - after_len}")


    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:   
                    messages = self._build_messages(doc)

                    # [USIM] if speak_as is true, pass in the name
                    if isinstance(messages, str):
                        raw_prompt = messages
                    elif self.speak_as:
                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, speak_as=doc['extra_info']['name'], **self.apply_chat_template_kwargs
                        ) + self.additional_generation_prompt
                    else:
                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False
                        ) + self.additional_generation_prompt
                        
                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])          
            else:
                def doc2len(doc) -> int:
                    # TODO: USERLM WILDCHAT - need to filter similarly for consistency
                    if self.is_wildchat and (self.new_sys_prompt is not None or self.augment_with_hierarchies):
                        doc1 = copy.deepcopy(doc)  
                        messages = self._build_messages(doc1) 

                        # [USIM] Apply new_sys_prompt replacement during filtering
                        if self.new_sys_prompt is not None:
                            new_system_prompt = self._get_new_system_prompt(doc1, self.new_sys_prompt_text)
                            if new_system_prompt is not None:
                                if isinstance(messages, list):
                                    messages = self._replace_system_prompt(messages, new_system_prompt)
                                elif isinstance(messages, str):
                                    messages = self._replace_qwen3_system(messages, new_system_prompt)
                        elif self.augment_with_hierarchies:
                            hierarchy_system_prompt = self._get_hierarchy_system_prompt(doc1, "response")
                            if hierarchy_system_prompt is not None:
                                if isinstance(messages, list):
                                    messages = self._replace_system_prompt(messages, hierarchy_system_prompt)
                                elif isinstance(messages, str):
                                    messages = self._replace_qwen3_system(messages, hierarchy_system_prompt)
                        if self.speak_as:
                            return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, speak_as=doc['extra_info']['name'], **self.apply_chat_template_kwargs)) + len(tokenizer.encode(self.additional_generation_prompt, add_special_tokens=False))
                        else:
                            return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **self.apply_chat_template_kwargs)) + len(tokenizer.encode(self.additional_generation_prompt, add_special_tokens=False))
                    else:
                        # [USIM] pass in name
                        if self.speak_as:
                            return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, speak_as=doc['extra_info']['name'], **self.apply_chat_template_kwargs)) + len(tokenizer.encode(self.additional_generation_prompt, add_special_tokens=False))
                        else:
                            return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs)) + len(tokenizer.encode(self.additional_generation_prompt, add_special_tokens=False))
            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
            self._original_len = len(self.dataframe)
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        if self.augment_with_hierarchies and self.is_train:
            return self._original_len * len(self.hierarchy_names)
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        # [USIM]
        if isinstance(messages, str):
            return messages

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    # ====== [USIM] replace system prompts when prompt is a string already
    def _replace_qwen3_system(self, prompt: str, new_system: str) -> str:
        if prompt.startswith(SYS_START):
            end = prompt.find(SYS_END, len(SYS_START))
            if end == -1:
                raise ValueError("No system prompt start found in prompt")

            old_system_block = prompt[len(SYS_START):end]  # everything inside system message
            suffix = prompt[end + len(SYS_END):]

            return SYS_START + new_system + SYS_END + suffix
        else:
            raise ValueError("no system prompt in prompt to replace")


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # row_dict: dict = self.dataframe[item]
        # messages = self._build_messages(row_dict)
        # model_inputs = {}

       # ===== [USIM] augmented indexing ===
        if ((self.augment_with_hierarchies and self.hierarchy_names) and self.is_train) or self.generate_hierarchies:
            original_idx = item % self._original_len
            hierarchy_idx = item // self._original_len
            hierarchy_name = self.hierarchy_names[hierarchy_idx]
        elif self.augment_with_hierarchies and not self.is_train:
            # if we are augmenting, during validation we only want to do on response-only prompts
            original_idx = item
            hierarchy_name = self.eval_hierarchy_name
        else:
            original_idx = item
            hierarchy_name = None
        
        row_dict: dict = self.dataframe[original_idx] 
        messages = self._build_messages(row_dict)
        model_inputs = {}

        # ===== [USIM] Apply hierarchy system prompt =====
        if self.augment_with_hierarchies and hierarchy_name:
            # Ensure extra_info exists
            if "extra_info" not in row_dict or row_dict["extra_info"] is None:
                row_dict["extra_info"] = {}

            if not isinstance(row_dict, dict):
                raise ValueError("ROW DICT not dict")
            
            if not isinstance(row_dict["extra_info"], dict):
                raise ValueError("Extra info not dict")
            
            row_dict["extra_info"]["hierarchy_name"] = hierarchy_name
            # print("============================")
            # print('HIERARCHY NAME: ', hierarchy_name)
            
            hierarchy_system_prompt = self._get_hierarchy_system_prompt(row_dict, hierarchy_name)
            if hierarchy_system_prompt is None:
                raise ValueError(f"Failed to construct hierarchy system prompt for '{hierarchy_name}'")
            if isinstance(messages, list):
                messages = self._replace_system_prompt(messages, hierarchy_system_prompt)
            elif isinstance(messages, str):
                messages = self._replace_qwen3_system(messages, hierarchy_system_prompt)
        
        if self.new_sys_prompt is not None:
            template = self.new_sys_prompt_text
            new_system_prompt = self._get_new_system_prompt(row_dict, template)

            if isinstance(messages, list):
                messages = self._replace_system_prompt(messages, new_system_prompt)
            elif isinstance(messages, str):
                messages = self._replace_qwen3_system(messages, new_system_prompt)
            # print("================================")
            # print(messages)
            # print("================================")
            
        if not self.generate_hierarchies and not self.is_train:
            row_dict["extra_info"]["hierarchy_name"] = "response"

        # ========

        if self.is_train and (self.field_dropout_prob > 0.0 or self.item_dropout_prob):
            messages = self._persona_dropout(row_dict, messages)

        if self.enable_hetero_think and self.is_train and not hierarchy_name == "response":
            apply_chat_template_kwargs = copy.deepcopy(self.apply_chat_template_kwargs)
            apply_chat_template_kwargs['enable_thinking'] = False
        else:
            apply_chat_template_kwargs = self.apply_chat_template_kwargs

         # ========
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            # [USIM] pass in name
            if isinstance(messages, str):
                raw_prompt = messages
            elif self.speak_as:
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, speak_as=row_dict['extra_info']['name'], tokenize=False, **apply_chat_template_kwargs) + self.additional_generation_prompt
            else:
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs) + self.additional_generation_prompt

            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            
            if isinstance(messages, str):
                raw_prompt = messages
            else:
                raw_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, speak_as=row_dict['extra_info']['name'], **apply_chat_template_kwargs
                ) + self.additional_generation_prompt
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()