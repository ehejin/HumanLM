# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import torch

from verl.workers.config.rollout import RolloutConfig, SamplingConfig
from verl.workers.rollout.vllm_rollout.logits_processors import NoRepeatNGramLogitsProcessor


def test_no_repeat_ngram_config():
    """Test that config classes support no_repeat_ngram_size parameter."""
    # Test SamplingConfig
    sampling_config = SamplingConfig(
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        no_repeat_ngram_size=3
    )
    
    assert sampling_config.no_repeat_ngram_size == 3
    assert sampling_config.temperature == 0.8
    
    # Test default value
    default_config = SamplingConfig()
    assert default_config.no_repeat_ngram_size == 0
    
    # Test RolloutConfig has the parameter
    assert hasattr(RolloutConfig, '__dataclass_fields__')
    assert 'no_repeat_ngram_size' in RolloutConfig.__dataclass_fields__
    
    print("✓ Config tests passed")


def test_no_repeat_ngram_processor():
    """Test NoRepeatNGramLogitsProcessor functionality."""
    # Test initialization
    processor = NoRepeatNGramLogitsProcessor(ngram_size=3)
    assert processor.ngram_size == 3
    
    # Test with short sequence (should not modify scores)
    prompt_tokens = (1, 2)
    past_tokens = ()
    vocab_size = 10
    scores = torch.randn(1, vocab_size)
    original_scores = scores.clone()
    
    processed = processor(prompt_tokens, past_tokens, scores)
    assert torch.equal(processed, original_scores)
    
    # Test with longer sequence (should modify scores)
    prompt_tokens = (1, 2, 3)
    past_tokens = (4, 5, 2, 3)
    scores = torch.randn(1, vocab_size)
    
    processed = processor(prompt_tokens, past_tokens, scores)
    assert processed.shape == scores.shape
    assert not torch.equal(processed, scores)
    
    # Test validation
    try:
        NoRepeatNGramLogitsProcessor(ngram_size=0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✓ Processor tests passed")


if __name__ == "__main__":
    test_no_repeat_ngram_config()
    test_no_repeat_ngram_processor()
    print("\nAll tests passed successfully!")
