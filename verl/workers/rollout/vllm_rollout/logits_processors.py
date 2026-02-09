# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Custom logits processors for vLLM rollout.
Implements functionality not natively available in vLLM SamplingParams.
"""

import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Logits processor to prevent n-gram repetitions during generation.
    
    This processor identifies n-grams that have already appeared in the generated sequence
    and sets their logits to negative infinity, preventing the model from repeating them.
    
    Args:
        ngram_size (int): Size of n-grams to track. Must be a positive integer.
                         For example, ngram_size=3 prevents any 3-word sequence from repeating.
    
    Reference:
        - https://github.com/vllm-project/vllm/issues/757
        - https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
    """
    
    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        self.ngram_size = ngram_size

    def __call__(
        self, 
        prompt_tokens_ids: tuple, 
        past_tokens_ids: tuple, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to prevent n-gram repetitions.
        
        Args:
            prompt_tokens_ids (tuple): Original prompt token IDs
            past_tokens_ids (tuple): Previously generated token IDs
            scores (torch.FloatTensor): Current logits scores [batch_size, vocab_size]
        
        Returns:
            torch.FloatTensor: Modified scores with banned n-gram tokens set to -inf
        
        Reference:
            https://github.com/vllm-project/vllm/blob/911c8eb0000b1f9d1fef99ac9e209f83d801bd0a/vllm/model_executor/layers/logits_processor.py#L186
        """
        # Combine prompt and generated tokens
        input_ids = prompt_tokens_ids + past_tokens_ids
        
        # If sequence is shorter than n-gram size, no need to process
        if len(input_ids) < self.ngram_size:
            return scores

        # Ensure scores have batch dimension
        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        num_batch_hypotheses = scores.shape[0]
        input_ids = torch.LongTensor(input_ids).reshape(num_batch_hypotheses, -1)
        cur_len = input_ids.shape[-1]
        
        # Clone scores to avoid in-place modification
        scores_processed = scores.clone()
        
        # Calculate which tokens would create banned n-grams
        banned_batch_tokens = _calc_banned_ngram_tokens(
            self.ngram_size, input_ids, num_batch_hypotheses, cur_len
        )
        
        # Set banned tokens to -inf
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed
