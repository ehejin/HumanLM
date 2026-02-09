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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
# near imports
import asyncio
import copy
import json
import os
import random
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.reward import (compute_reward, compute_reward_async,
                                     compute_reward_with_reasoning)
from verl.trainer.ppo.utils import (Role, WorkerType, need_critic,
                                    need_reference_policy, need_reward_model)
from verl.utils.checkpoint.checkpoint_manager import (find_latest_ckpt_path,
                                                      should_save_ckpt_esi)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import (get_seqlen_balanced_partitions,
                                         log_seqlen_unbalance)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PRM:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_process_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def interleaved_concatenate_dataproto(dataproto_1, dataproto_2):
    # Merge two DataProto in a way that 
    # if gen_batch_output is [1,1,1,2,2,2,3,3,3], gd_batch_output is [1', 2', 3']
    # the merged DataProto should be [1,1,1,1',2,2,2,2',3,3,3,3']
    
    # Calculate samples per ground truth
    size_1 = dataproto_1.batch.batch_size[0]
    size_2 = dataproto_2.batch.batch_size[0]
    samples_per_gt = size_1 // size_2
    
    # Create merged tensors
    merged_batch = {}
    for key in dataproto_1.batch.keys():
        gen_tensor = dataproto_1.batch[key]
        gt_tensor = dataproto_2.batch[key]
        
        merged_list = []
        for i in range(size_2):
            merged_list.append(gen_tensor[i * samples_per_gt:(i + 1) * samples_per_gt])
            merged_list.append(gt_tensor[i:i + 1])
        merged_batch[key] = torch.cat(merged_list, dim=0)
    
    # Create merged non_tensor_batch
    merged_non_tensor = {}
    for key in dataproto_1.non_tensor_batch.keys():
        gen_array = dataproto_1.non_tensor_batch[key]
        gt_array = dataproto_2.non_tensor_batch[key]
        
        merged_list = []
        for i in range(size_2):
            # Add generated samples
            merged_list.extend(gen_array[i * samples_per_gt:(i + 1) * samples_per_gt])
            # Add ground truth sample
            merged_list.append(gt_array[i])
        
        merged_non_tensor[key] = np.array(merged_list, dtype=object)
    
    meta_info = deepcopy(dataproto_1.meta_info)
    meta_info.update(dataproto_2.meta_info)
    merged_data = DataProto(
        batch=TensorDict(merged_batch, batch_size=[size_1 + size_2]),
        non_tensor_batch=merged_non_tensor,
        meta_info=meta_info
    )
    return merged_data
    
    
class HierarchyTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        # [DIST]
        self._val_dist_counter = 0

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        if self.config.trainer.get("concatenate_hierarchy_batches", False):
            assert not self.config.trainer.get("interleave_ground_truth", False)
            assert not self.config.trainer.get("separate_generation", False)
            assert self.config.data.get("additional_generation_prompt", '') == ''

        self.augment_with_hierarchies = self.config.data.get("augment_with_hierarchies", False)
        self.separate_rewards = self.config.reward_model.reward_kwargs.get("separate_rewards", False)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import \
                collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )


        # assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        # assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch


    def _validate(self):
        
        # [PANGRAM]
        if self.val_reward_fn.pangram_enabled:
            self.val_reward_fn.reset_pangram_accumulator()
        
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        val_reward_weighted_sum = defaultdict(float)  
        val_reward_weighted_cnt = defaultdict(int)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_reward_dicts = []

        #[DIST]
        all_extra_infos = []
        dist_texts_all = []
        self._val_dist_counter += 1

        distribution_eval_freq = int(self.config.trainer.get("distribution_eval_freq", 1))
        run_dist_now = self.config.trainer.do_distribution_eval and (self._val_dist_counter % distribution_eval_freq == 0)

        hierarchy_metrics = {h: {"scores": [], "count": 0} for h in self.hierarchy_names} if self.augment_with_hierarchies else {}

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # [DIST]
            # extract the last hierarchy from the model generations for metric scoring
            # if run_dist_now:
            #     assert ValueError("To check")
            #     from verl.workers.reward_manager.usim import parse_fields
            #     import re

            #     last_hierarchy_name = self.hierarchy_names[-1]

            #     last_hierarchy_only_texts = []
            #     for txt in output_texts:
            #         fields = parse_fields(txt)  # take last occurence if tag repeats
            #         frag = fields.get(last_hierarchy_name.lower())
            #         last_hierarchy_only_texts.append(frag)
                
            #     per_output_extra_infos = [
            #         item.non_tensor_batch.get("extra_info", {}) for item in test_batch
            #     ]

            #     assert len(last_hierarchy_only_texts) == len(per_output_extra_infos) 
            #     dist_texts_all.extend(last_hierarchy_only_texts)
            #     all_extra_infos.extend(per_output_extra_infos)

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)

            # [USIM] Edit for wandb logging
            batch_n = len(test_batch.batch["input_ids"])
            if "reward_extra_info" in result:
                for k, v in result["reward_extra_info"].items():
                    try:
                        v = float(v.item() if hasattr(v, "item") else v)
                    except:
                        continue
                    val_reward_weighted_sum[k] += v * batch_n
                    val_reward_weighted_cnt[k] += batch_n
            
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            sample_scores.extend(scores)
            sample_reward_dicts.extend(result['reward_dicts'])
            extra = result.pop('reward_extra_info', None)

            if self.augment_with_hierarchies:
                batch_hierarchy_names = [
                    item.non_tensor_batch.get("extra_info", {}).get("hierarchy_name", "unknown")
                    for item in test_batch
                ]
                for i, (h_name, score) in enumerate(zip(batch_hierarchy_names, scores)):
                    if h_name in hierarchy_metrics:
                        hierarchy_metrics[h_name]["scores"].append(score)
                        hierarchy_metrics[h_name]["count"] += 1

            reward_extra_infos_dict["reward"].extend(scores)
            # if isinstance(extra, dict):
            #     for key, lst in extra.items():
            #         reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        metric_dict = {}
        if self.val_reward_fn.pangram_enabled:
            pangram_metrics, windows = asyncio.run(self.val_reward_fn.finalize_pangram())
            metric_dict.update(pangram_metrics)
            
            if 'wandb' in self.config.trainer.logger:
                try:
                    import wandb
                    if wandb.run is not None and windows:
                        cols = []
                        for w in windows:
                            for k in w.keys():
                                if k not in cols:
                                    cols.append(k)

                        table = wandb.Table(columns=["global_step"] + cols)
                        for w in windows:
                            table.add_data(self.global_steps, *[w.get(c) for c in cols])

                        wandb.log({"val/pangram_windows": table}, step=self.global_steps)
                except Exception as e:
                    print(f"[WARN] failed to log pangram windows table to wandb: {e}")

        sample_outputs_with_gt = [json.dumps({'output': output, "ground_truth": gt}) for output, gt in zip(sample_outputs, sample_gts)]
        sample_scores_to_log = [json.dumps({"score": score, **score_dict}) for score, score_dict in zip(sample_scores, sample_reward_dicts)]
        
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs_with_gt, scores=sample_scores_to_log)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )
        
        # push to hub
        if self.val_reward_fn.eval_push_to_hub:
            keys = list(self.val_reward_fn.eval_cache.keys())
            hf_dataset = Dataset.from_dict(
                {
                    "key": keys,
                    **{
                        k: [self.val_reward_fn.eval_cache[key][k] for key in keys]
                        for k in self.val_reward_fn.eval_cache[keys[0]].keys()
                    },
                }
            )
            hf_dataset.push_to_hub(
                self.val_reward_fn.eval_push_to_hub,
                split=f'step_{self.global_steps}',
                private=True,
            )

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)

        #[USIM]
        for k, s in val_reward_weighted_sum.items():
            cnt = val_reward_weighted_cnt[k]
            if cnt > 0:
                metric_dict[f"val-aux/{k}"] = s / cnt

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if self.augment_with_hierarchies:
            for h_name, h_data in hierarchy_metrics.items():
                if h_data["scores"]:
                    metric_dict[f"val/{h_name}/reward/mean"] = np.mean(h_data["scores"])
                    metric_dict[f"val/{h_name}/reward/std"] = np.std(h_data["scores"])
                    metric_dict[f"val/{h_name}/count"] = h_data["count"]

        # [DIST]
        # Agg TVD stats across all posts in val run
        if run_dist_now:
            metric_updates = self._compute_post_level_tvd(dist_texts_all, all_extra_infos)
            metric_dict.update(metric_updates)
        
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # [LLM_TWIN] save dataloader and hierarchy map inside actor path
        target_dir = actor_remote_path or actor_local_path
        local_mkdir_safe(target_dir)
        dataloader_local_path = os.path.join(target_dir, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # [LLM_TWIN] save the global hierarchy map
        hierarchy_map_path = os.path.join(target_dir, "hierarchy_map.pt")
        hierarchy_map = self.reward_fn.state_dict()
        print("==============================")
        print("[SAVED HIERARCHY MAP] ", hierarchy_map)
        print("==============================")
        torch.save(hierarchy_map, hierarchy_map_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")

        dl_path = os.path.join(actor_path, "data.pt")
        if not os.path.exists(dl_path):
            dl_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dl_path):
            dataloader_state_dict = torch.load(dl_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            raise Exception(f"[ERROR]: No dataloader state found at {dl_path}, will start from scratch")
        if self.config.trainer.load_hierarchy_map:
            # TODO: from remote not implemented yet    
            # [LLM_TWIN] IF HIERARCHY TRAINING load global hierarchy map
            reward_state_path = os.path.join(actor_path, "hierarchy_map.pt") 
            if not os.path.exists(reward_state_path):
                reward_state_path = os.path.join(global_step_folder, "hierarchy_map.pt") 
            if not os.path.exists(reward_state_path):
                raise Exception("[ERROR] Hierarchy map missing from checkpoint")
            reward_state = torch.load(reward_state_path, map_location="cpu", weights_only=False)
            self.reward_fn.load_state_dict(reward_state)

        # load actor (moved to end in case del_local_after_load)
        print(f"Loading actor from {actor_path}")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        
        # load critic
        print(f"Loading critic from {critic_path}")
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )


    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
    
    def _get_trainset_list(self, idx: int):
        if idx == 0:
            return self.train_datasets[0]
        last_hierarchy = self.hierarchy_names[idx - 1]
        cur_hierarchy = self.hierarchy_names[idx]
        key_to_data_list = self.reward_fn.hierarchy_map.get(last_hierarchy, None)

        if key_to_data_list is None:
            return []

        # only take the first data for current hierarchy
        train_dataset_list = [
            data_list[0]["data"]
            for data_list in key_to_data_list.values()
        ]
        return train_dataset_list

    def _update_train_dataset(self, step: int, idx: int):
        import os
        from pathlib import Path

        import polars as pl

        from verl.utils.dataset.rl_dataset import \
            collate_fn as default_collate_fn

        cur_hierarchy = self.hierarchy_names[idx]
        train_dataset_list = self._get_trainset_list(idx)

        print('TDL++++', train_dataset_list)
        print(step, idx)

        #assert len(train_dataset_list) > 0
        if len(train_dataset_list) == 0:
            print(f"[Step {step}] No data found for hierarchy '{cur_hierarchy}' (idx={idx}). Skipping dataset update.")
            return
        s0 = train_dataset_list[0]

        if len(train_dataset_list) == 0:
            print(f"No data found for hierarchy {cur_hierarchy} at step {step}. Skipping dataset update.")
            return
        
        # Save filtered dataset
        cache_dir = Path(self.config.trainer.default_local_dir) / "train_dataset"
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / f"{cur_hierarchy}.parquet"

        def sanitize_for_parquet(df: pl.DataFrame) -> pl.DataFrame:
            sanitized = {}
            for col in df.columns:
                s = df[col]
                # Replace struct[0] columns with None
                if s.dtype == pl.Struct and len(s.struct.fields) == 0:
                    sanitized[col] = pl.Series(name=col, values=[None] * len(s))
                else:
                    sanitized[col] = s
            return pl.DataFrame(sanitized)

        df = pl.DataFrame(train_dataset_list)
        df = sanitize_for_parquet(df)
        df.write_parquet(file_path)
        print(f"Saved {len(df)} samples to {file_path}")

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        train_dataset = create_rl_dataset(
            [str(file_path)],
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
        )

        # Register new dataset and dataloader
        import copy
        data_config = copy.deepcopy(self.config.data)
        data_config.shuffle = True
        data_config.seed = step

        # have sampler here to account for the imbalanced
        # data now we put in hierarchy map is above the thresold => good data
        
        # We use the previous hierarchies to inform the sampling of the next hierarchy
        # for example, 
        # for 2nd hierarchy:
        # stance_list = ['agree', 'disagree', 'agree'] for the same post
        # weight      = [0.5, 1, 0.5]
        # for 3rd hierarchy:
        # stance_list   = ['agree', 'disagree', 'agree']
        # sentiment_list = ['positive', 'positive', 'negative']
        # combination = ['agree|positive', 'disagree|positive', 'agree|negative']
        # weight         = [1,1,1]

        self.train_datasets[idx] = train_dataset
        self.train_dataloaders[idx] = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 4),
            sampler=create_rl_sampler(data_config, train_dataset),
            collate_fn=default_collate_fn,
            drop_last=True,
        )
        self.train_iterators[idx] = iter(self.train_dataloaders[idx])


    def append_outputs_to_dataproto(self, data: DataProto, outputs: list[str], hierarchy_names: list[str]):
        from verl.utils.torch_functional import (get_response_mask,
                                                 pad_2d_list_to_length)

        # this needs further check, 
        # in vllm's output gen_batch, no eos token found, but not sure if this is always true
        add_eos_token = False 

        assert len(outputs) == len(data.batch["input_ids"])
        
        response_lst = []
        for hierarchy_name, out in zip(hierarchy_names, outputs):

            if self.config.data.get("additional_generation_prompt", ''):
                _open = f"\n"
            else:
                _open = f"<{hierarchy_name}>\n"
            _close = f"\n</{hierarchy_name}>"

            wrapper_ids = self.tokenizer.encode(_open + _close, add_special_tokens=False)
            wrapper_len = len(wrapper_ids)
            max_content_len = max(0, self.config.data.max_response_length - wrapper_len - 1)

            resp_ids = self.tokenizer.encode(out, add_special_tokens=False)
            resp_ids = resp_ids[:max_content_len]

            response_lst.append(
                self.tokenizer.encode(_open, add_special_tokens=False)
                + resp_ids
                + self.tokenizer.encode(_close, add_special_tokens=False)
                + ([self.tokenizer.eos_token_id] if add_eos_token else [])
            )
        
        idx = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        non_tensor_batch = data.non_tensor_batch
        non_tensor_batch.pop("raw_prompt_ids", None)
        batch_size = idx.size(0)

        response = pad_2d_list_to_length(response_lst, self.tokenizer.pad_token_id, max_length=self.config.data.max_response_length).to(
            idx.device
        )
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=self.tokenizer.eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    
    def _truncate_generations_to_hierarchy(self, gen_batch_output: DataProto, hierarchy_names_per_sample: list[str]) -> DataProto:
        """Truncate each sample to its own target hierarchy."""
        responses = gen_batch_output.batch["responses"]
        attention_mask = gen_batch_output.batch["attention_mask"]
        prompts = gen_batch_output.batch["prompts"]
        
        batch_size = responses.size(0)
        prompt_length = prompts.size(1)
        
        for i in range(batch_size):
            hierarchy_name = hierarchy_names_per_sample[i]
            closing_tag = f"</{hierarchy_name}>"
            
            response_attention = attention_mask[i, prompt_length:]
            actual_length = int(response_attention.sum().item())
            response_ids = responses[i, :actual_length].tolist()
            
            decoded_response = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            closing_pos = decoded_response.find(closing_tag)
            
            if closing_pos != -1:
                truncated_text = decoded_response[:closing_pos + len(closing_tag)]
                truncated_ids = self.tokenizer.encode(truncated_text, add_special_tokens=False)
                truncate_length = len(truncated_ids)
                
                if truncate_length < actual_length:
                    attention_mask[i, prompt_length + truncate_length:] = 0
                    responses[i, truncate_length:] = self.tokenizer.pad_token_id
        
        gen_batch_output.batch["responses"] = responses
        gen_batch_output.batch["attention_mask"] = attention_mask
        return gen_batch_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        self.separate_generation = self.config.reward_model.reward_kwargs.get("separate_generation", False)
        self.enable_hierarchy = self.config.reward_model.reward_kwargs.get("enable_hierarchy", False)
        self.hierarchy_config = json.loads(open(self.config.reward_model.reward_kwargs.hierarchy_config).read())
        self.hierarchy_names = list(self.hierarchy_config.keys())


        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                logger.finish()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        

        self.train_dataloaders = {0: self.train_dataloader}
        self.train_iterators = {0: iter(self.train_dataloader)}
        self.train_datasets = {0: self.train_dataset}
        avg_reward_per_hierarchy = {name: 1. for name in self.hierarchy_names}
        
        batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        late_update_set: set[int] = set() if self.separate_generation or self.separate_rewards else set(range(1, len(self.hierarchy_names)))

        for step in range(self.total_training_steps):
            if self.augment_with_hierarchies:
                # Just iterate through the single augmented dataloader
                try:
                    batch_dict = next(self.train_iterators[0])
                except StopIteration:
                    self.train_iterators[0] = iter(self.train_dataloaders[0])
                    batch_dict = next(self.train_iterators[0])
                
                # For logging - count hierarchies in batch
                batch_hierarchy_names = [
                    info.get("hierarchy_name") 
                    for info in batch_dict.get("extra_info", [{}] * len(batch_dict["input_ids"]))
                ]
                hierarchy_counts = Counter(batch_hierarchy_names)
                print(f"[Step {step}] Augmented batch hierarchy distribution: {dict(hierarchy_counts)}")
                
                # These are used later in the loop
                hierarchy_name = "mixed"  # For logging
                idx = 0
                is_last_hierarchy = False  # Not applicable in augmented mode
                batch_size_dict = hierarchy_counts

            # Compute sampling weights by current trainset sizes at each hierarchy
            elif self.enable_hierarchy and self.config.trainer.get("concatenate_hierarchy_batches", False):
                # Concatenate batches from all hierarchy levels
                batches = []
                batch_size_dict = {}
                
                # by importance sampling, we set micro batch size
                prob = torch.nn.functional.softmax(
                    torch.tensor([-avg_reward_per_hierarchy[h] for h in self.hierarchy_names]), 
                    dim=0
                )
                planned_batch_size_dict = {h: int(batch_size * prob[i]) for i, h in enumerate(self.hierarchy_names)}
                print(f'prob {prob} | planned_batch_size_dict {planned_batch_size_dict}')
                
                # Iterate from highest hierarchy to lowest (reverse order)
                for idx in range(len(self.hierarchy_names) - 1, -1, -1):
                    hierarchy_name = self.hierarchy_names[idx]
                    is_last_hierarchy = idx == (len(self.hierarchy_names) - 1)
                    is_first_hierarchy = idx == 0
                    
                    # Check if this hierarchy has enough data
                    size = len(self._get_trainset_list(idx=idx))
                    micro_batch_size = planned_batch_size_dict[hierarchy_name]
                    if size < micro_batch_size:
                        continue
                    
                    # Schedule late update for next level if exists
                    if not is_last_hierarchy:
                        late_update_set.add(idx + 1)

                    # Get a batch from this hierarchy
                    try:
                        batch_dict = next(self.train_iterators[idx])
                    except StopIteration:
                        try:
                            self._update_train_dataset(step=step, idx=idx)
                            self.train_iterators[idx] = iter(self.train_dataloaders[idx])
                            batch_dict = next(self.train_iterators[idx])
                        except StopIteration:
                            continue
                    except Exception as e:
                        continue
                    
                    # Sample from the batch
                    if is_first_hierarchy:
                        # Root hierarchy (idx == 0): take enough to fill up to batch_size
                        total_added = sum(len(batch[list(batch.keys())[0]]) for batch in batches)
                        remaining_size = batch_size - total_added
                        
                        import numpy as np
                        first_key = list(batch_dict.keys())[0]
                        batch_len = len(batch_dict[first_key])
                        sample_size = min(remaining_size, batch_len)
                        indices = np.random.permutation(batch_len)[:sample_size]
                        
                        root_batch = {}
                        for key in batch_dict.keys():
                            if isinstance(batch_dict[key], torch.Tensor):
                                root_batch[key] = batch_dict[key][indices]
                            elif isinstance(batch_dict[key], np.ndarray):
                                root_batch[key] = batch_dict[key][indices]
                            else:
                                # Handle lists or other sequences
                                root_batch[key] = [batch_dict[key][i] for i in indices]
                        batches.append(root_batch)
                    else:
                        # Randomly sample micro_batch_size from non-root hierarchy
                        first_key = list(batch_dict.keys())[0]
                        batch_len = len(batch_dict[first_key])
                        sample_size = min(micro_batch_size, batch_len)
                        indices = np.random.permutation(batch_len)[:sample_size]
                        
                        micro_batch = {}
                        for key in batch_dict.keys():
                            if isinstance(batch_dict[key], torch.Tensor):
                                micro_batch[key] = batch_dict[key][indices]
                            elif isinstance(batch_dict[key], np.ndarray):
                                micro_batch[key] = batch_dict[key][indices]
                            else:
                                # Handle lists or other sequences
                                micro_batch[key] = [batch_dict[key][i] for i in indices]
                        batches.append(micro_batch)
                    
                    batch_size_dict[hierarchy_name] = sample_size

                # Concatenate all collected batches
                if batches:
                    if len(batches) == 1:
                        batch_dict = batches[0]
                    else:
                        combined_batch = {}
                        for key in batches[0].keys():
                            if key == 'uid':
                                continue
                            if isinstance(batches[0][key], torch.Tensor):
                                combined_batch[key] = torch.cat([batch[key] for batch in batches], dim=0)
                            elif isinstance(batches[0][key], np.ndarray):
                                combined_batch[key] = np.concatenate([batch[key] for batch in batches], axis=0)
                            else:
                                # Handle lists - flatten into single list
                                combined_batch[key] = [item for batch in batches for item in batch[key]]
                        batch_dict = combined_batch
                        hierarchy_name = f"concatenated_hierarchies"
                else:
                    continue

                pre_repeat_hierarchy_names = [
                    info.get("hierarchy_name", self.hierarchy_names[0])
                    for info in batch_dict.get("extra_info", [{}] * len(batch_dict[list(batch_dict.keys())[0]]))
                ]
            # Sample a single hierarchy level based on data availability
            else:
                sizes = [
                    len(self._get_trainset_list(idx=i))
                    for i, name in enumerate(self.hierarchy_names)
                ] 
                print("--------------------------------------------------")
                print("SIZES", sizes)

                weights = [float(s >= batch_size) for s in sizes]
                print("WEIGHTS", weights)
                print("--------------------------------------------------")

                idx = random.choices(range(len(self.hierarchy_names)), weights=weights, k=1)[0]
                is_last_hierarchy = idx == (len(self.hierarchy_names) - 1)
                hierarchy_name = self.hierarchy_names[idx]

                # If we sampled level k, schedule a late update for level k+1 (if it exists)
                if not is_last_hierarchy:
                    late_update_set.add(idx + 1)

                # Get a batch from the (possibly refreshed) iterator
                try: 
                    batch_dict = next(self.train_iterators[idx])
                except (StopIteration, KeyError):
                    try:
                        self._update_train_dataset(step=step, idx=idx)
                        self.train_iterators[idx] = iter(self.train_dataloaders[idx])
                        batch_dict = next(self.train_iterators[idx])
                    except StopIteration:
                        continue
                
                batch_size_dict = {hierarchy_name: len(batch_dict[list(batch_dict.keys())[0]])}
                pre_repeat_hierarchy_names = [
                    info.get("hierarchy_name", self.hierarchy_names[0])
                    for info in batch_dict.get("extra_info", [{}] * len(batch_dict[list(batch_dict.keys())[0]]))
                ]

            print(f"[Step {step}] Target={hierarchy_name} | "
                  f"Total sizes={[len(self._get_trainset_list(i)) for i in range(len(self.hierarchy_names))]} | "
                  f"Sample sizes={batch_size_dict} | "
                  f"late_update_set={sorted(late_update_set)}")

            if step > 0 and step % self.config.trainer.get("update_hierarchy_loader_freq", 1) == 0:
                for idx in late_update_set:
                    self._update_train_dataset(step=step, idx=idx)
                late_update_set = set()

            metrics = {}
            timing_raw = {}
            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
            batch: DataProto = DataProto.from_single_dict(batch_dict)


            import numpy as np

            # add uid to batch
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = self.global_steps
            # we set stop here, which is used in vllm_rollout_spmd.py
            if self.separate_generation:
                if self.augment_with_hierarchies:
                        gen_batch.meta_info.update({
                        "stop": [f"</{h}>" for h in self.hierarchy_names],
                    })
                else:
                    gen_batch.meta_info.update({
                        "stop": [f"</{hierarchy_name}>"],
                        "max_tokens": self.hierarchy_config[hierarchy_name]["max_tokens"],
                    })
            if self.augment_with_hierarchies:
                assert self.separate_generation is True

            n = self.config.actor_rollout_ref.rollout.n
            if is_last_hierarchy and self.config.trainer.get("interleave_ground_truth", False): 
                n = n - 1

            non_repeat_gen_batch = copy.deepcopy(gen_batch)
            gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)

            if self.separate_rewards and pre_repeat_hierarchy_names is not None:
                batch_hierarchy_names_repeated = []
                for name in pre_repeat_hierarchy_names:
                    batch_hierarchy_names_repeated.extend([name] * n)
            
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw, color="red"):
                    if not self.async_rollout_mode:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    else:
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        # TODO: cut off after current hierarchy closing tag here
                    

                    if self.separate_rewards:
                        if self.config.trainer.get("concatenate_hierarchy_batches", False) or self.augment_with_hierarchies:
                            assert batch_hierarchy_names_repeated is not None, \
                                "batch_hierarchy_names_repeated should be set for concatenate_hierarchy_batches"
                            assert len(batch_hierarchy_names_repeated) == len(gen_batch_output.batch["input_ids"]), \
                                f"Hierarchy names {len(batch_hierarchy_names_repeated)} != gen_batch_output size {len(gen_batch_output.batch['input_ids'])}"
                            
                            gen_batch_output = self._truncate_generations_to_hierarchy(
                                gen_batch_output, batch_hierarchy_names_repeated
                            )
                        else:
                            # Single hierarchy 
                            single_hierarchy_names = [hierarchy_name] * len(gen_batch_output.batch["input_ids"])
                            gen_batch_output = self._truncate_generations_to_hierarchy(
                                gen_batch_output, single_hierarchy_names
                            )


                    if self.config.trainer.get("interleave_ground_truth", False) and is_last_hierarchy:
                        # encode ground truth with tokenizer and insert to gen_batch_output, this will receive a reward of 1.0
                        ground_truth = [item["ground_truth"] for item in batch.non_tensor_batch["reward_model"]]
                        hierarchy_names = [hierarchy_name] * len(ground_truth)
                        ground_truth_batch_output = self.append_outputs_to_dataproto(non_repeat_gen_batch, ground_truth, hierarchy_names)
                        gen_batch_output = interleaved_concatenate_dataproto(gen_batch_output, ground_truth_batch_output)

                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    if self.reward_fn is None:
                        raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                    with marked_timer("gen_max", timing_raw, color="purple"):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False

                        if not self.async_rollout_mode:
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                        else:
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                with marked_timer("reward", timing_raw, color="yellow"):
                    # compute reward model score
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    if self.config.reward_model.launch_reward_fn_async:
                        future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                    else:
                        #reward_tensor, llm_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        reward_tensor, llm_extra_infos_dict, reward_dicts_for_logging = compute_reward_with_reasoning(batch, self.reward_fn)
                        #print(f"[DEBUG STEP {self.global_steps}] llm_extra_infos_dict keys: {list(llm_extra_infos_dict.keys())}")
                        reward_extra_infos_dict = {}

                        if self.config.trainer.get("interleave_ground_truth", False) and is_last_hierarchy:
                            batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)

                            if sum(reward_tensor.sum(dim=-1) == 1.0) < batch_size:
                                print(f"Warning: Expected at least {batch_size} rewards of 1.0, got {sum(reward_tensor.sum(dim=-1) == 1.0)}")
                
                        # update average reward per hierarchy
                        if self.enable_hierarchy and not self.augment_with_hierarchies and self.config.trainer.get("concatenate_hierarchy_batches", False):
                            for h in self.hierarchy_names:
                                if self.separate_rewards:
                                    if self.config.trainer.get("concatenate_hierarchy_batches", False):
                                        metric_key = f"train/{h}:state_reward:score"
                                        if metric_key in llm_extra_infos_dict:
                                            avg_reward_per_hierarchy[h] = llm_extra_infos_dict[metric_key]
                                    elif h != hierarchy_name:
                                        continue
                                    else:
                                        avg_reward_per_hierarchy[h] = llm_extra_infos_dict[f"train/{h}:state_reward:score"]
                                else:
                                    avg_reward_per_hierarchy[h] = llm_extra_infos_dict[f"train/{h}:state_reward:score"]
                
                # [USIM] Logging
                if self.global_steps % self.config.trainer.save_freq == 0 or self.global_steps == 1:
                    n = self.config.actor_rollout_ref.rollout.n
                    num_to_log = min(self.config.trainer.get("num_train_examples", 3), len(batch.batch["prompts"]) // n)
                    
                    
                    # Use batch (after repeat/union) with [::n] to get one per original sample
                    train_inputs = self.tokenizer.batch_decode(
                        batch.batch["prompts"][::n][:num_to_log], 
                        skip_special_tokens=True
                    )
                    
                    train_outputs = self.tokenizer.batch_decode(
                        batch.batch["responses"][::n][:num_to_log],  
                        skip_special_tokens=True
                    )

                    train_scores = reward_tensor.sum(dim=-1)[::n][:num_to_log].tolist()
                    
                    # Get ground truth from batch.non_tensor_batch (which has reward_model)
                    train_gts = [
                        item["ground_truth"] for item in batch.non_tensor_batch["reward_model"][::n][:num_to_log]
                    ]

                    train_reasoning = []
                    for i in range(num_to_log):
                        idx1 = i * n
                        rd = reward_dicts_for_logging[idx1] if idx1 < len(reward_dicts_for_logging) else {}
                        
                        parts1 = []
                        for field1, metrics1 in rd.items():
                            if isinstance(metrics1, dict):
                                for key, val in metrics1.items():
                                    if "metrics_info" in key and val:
                                        parts1.append(f"[{field1}] {val}")
                        train_reasoning.append("\n".join(parts1) if parts1 else "")

                    if 'wandb' in self.config.trainer.logger:
                        import wandb
                        if wandb.run is not None:
                            table = wandb.Table(columns=["step", "input", "output", "ground_truth", "reward", "reasoning"])
                            for inp, out, gt, score, reason in zip(train_inputs, train_outputs, train_gts, train_scores, train_reasoning):
                                table.add_data(self.global_steps, inp, out, gt, score, reason[:3000])
                            wandb.log({"train/generation_samples": table}, step=self.global_steps)
                            
                # recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        from verl.utils.debug.metrics import \
                            calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get(
                        "norm_adv_by_std_in_grpo", True
                    )  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            # Save hierarchy map
            if not self.augment_with_hierarchies: 
                hierarchy_map = self.reward_fn.state_dict()
                if len(hierarchy_map) and self.config.trainer.save_hierarchy_map_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_hierarchy_map_freq == 0
                ):

                    # Transform your hierarchy_map into the format HF expects
                    dataset_dict = {}

                    for hierarchy_name, key_dict in hierarchy_map.items():
                        if len(key_dict) == 0:
                            continue
                        
                        rows = []
                        for key, value_list in key_dict.items():
                            for item in value_list:
                                row = {
                                    'key': key,
                                    'prompt': item['data']['prompt'],
                                    'ground_truth': item['data']['reward_model']['ground_truth'],
                                    'field': item['field'],
                                    'reward': item['reward'],
                                    'data': json.dumps(item['data']),
                                    'timestamp': item['timestamp']
                                }
                                rows.append(row)
                        
                        # Only add non-empty datasets
                        if rows:
                            dataset_dict[hierarchy_name] = Dataset.from_dict({
                                'key': [r['key'] for r in rows],
                                'prompt': [r['prompt'] for r in rows],
                                'ground_truth': [r['ground_truth'] for r in rows],
                                'field': [r['field'] for r in rows],
                                'reward': [r['reward'] for r in rows],
                                'data': [r['data'] for r in rows],
                                'timestamp': [r['timestamp'] for r in rows]
                            })

                    # Only push if we have at least one non-empty dataset
                    if dataset_dict:
                        hf_dataset = DatasetDict(dataset_dict)
                        hf_dataset.push_to_hub(f"hf-org/hmap_{self.config.trainer.experiment_name}", private=True)

            # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
            esi_close_to_expiration = should_save_ckpt_esi(
                max_steps_duration=self.max_steps_duration,
                redundant_time=self.config.trainer.esi_redundant_time,
            )
            # Check if the conditions for saving a checkpoint are met.
            # The conditions include a mandatory condition (1) and
            # one of the following optional conditions (2/3/4):
            # 1. The save frequency is set to a positive value.
            # 2. It's the last training step.
            # 3. The current step number is a multiple of the save frequency.
            # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
            ):
                if esi_close_to_expiration:
                    print("Force saving checkpoint: ESI instance expiration approaching.")
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # training metrics
            metrics.update(
                {
                    "training/hierarchy_idx": idx,
                    "training/global_step": self.global_steps,
                    "training/epoch": step // self.total_training_steps
                }
            )
            # [USIM] Edit for wandb logging
            if llm_extra_infos_dict:
                metrics.update(llm_extra_infos_dict)

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            # if isinstance(self.train_dataloaders[idx].sampler, AbstractCurriculumSampler):
            #     self.train_dataloaders[idx].sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if (
                hasattr(self.config.actor_rollout_ref.actor, "profiler")
                and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
            ):
                self.actor_rollout_wg.dump_memory_snapshot(
                    tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                )

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                logger.finish()
                return

            # this is experimental and may be changed/removed in the future
            # in favor of a general-purpose data buffer pool
            # if hasattr(self.train_datasets[idx], "on_batch_end"):
            #     # The dataset may be changed after each training batch
            #     self.train_datasets[idx].on_batch_end(batch=batch)

        # If we exit the loop without hitting a return (e.g., total_training_steps == 0)
        try:
            progress_bar.close()
        except Exception:
            pass
        logger.finish()