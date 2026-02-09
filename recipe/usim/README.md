# USIM Recipe for VERL

A custom recipe for training user simulators using the VERL reinforcement learning framework with hierarchical state alignment.
## Overview

**Key Insight:** Instead of simply imitating user responses, HumanLM aligns model generations with psychologically grounded **latent states** that drive how real users respond.

This recipe extends VERL with:
- **Hierarchical state alignment**: Generate and align latent states across multiple psychological dimensions before synthesizing the final response
- **Custom reward functions**: LLM-based judges (Claude Haiku, GPT-5) evaluate both latent states and responses
- **Flexible training modes**: Support for supervised fine-tuning (SFT), reinforcement learning (RL), and thinking trace generation

## Key Components

### Training Approach
1. **Generate latent states** for each dimension
2. **Score alignment** using LLM judge comparing generated states to ground truth
3. **Synthesize responses** by generating reasoning traces that integrate aligned latent states
4. **Optimize with RL** (GRPO) to maximize alignment scores on both states and responses

## Prerequisites

**Authentication:**
```bash
# Login to required services
wandb login
huggingface-cli login

# Export API keys for reward model evaluation
export OPENAI_API_KEY={YOUR_KEY}
export ANTHROPIC_API_KEY={YOUR_KEY}
```

## Installation
```bash
cd verl
git pull
```

Download base model:
```bash
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir $HOME/llm_twin/models/Qwen3-8B \
    --local-dir-use-symlinks False
```

## Data Processing

Use `create_any_dataset.py` to convert raw conversation data into VERL-compatible format with hierarchical structure:
```bash
# Basic usage
python -m recipe.usim.create_any_dataset \
    --dataset {DATASET_NAME} \
    --raw_dataset_repo {HF_REPO} \
    --save_data_dir {OUTPUT_DIR} \
    --save_prompt_dir ./recipe/usim/system_prompt/ \
    --hierarchy_config_path ./recipe/usim/hierarchy_config/sebvgcr.json \
    --train_subset_percentage 50
```

### Key Arguments

- `--dataset`: Dataset type (reddit, amazon, youtube, medium, wildchat_english, enron)
- `--hierarchy_config_path`: JSON file defining state dimensions and their descriptions
- `--sft`: Generate supervised fine-tuning data instead of RL data
- `--thinking_sft`: Generate synthetic thinking traces for SFT (requires `--sft`)
- `--thinking_model`: LLM to use for generating thinking traces (default: `gpt-4o-mini`)
- `--no_tag`: Remove XML tags from responses (for baseline models)

### Hierarchy Configuration

We provide a JSON file defining the state hierarchy used in the paper (`hierarchy_config/sebvgcr.json`):
```json
{
    "stance":
    {
        "desc": "HUMAN's agreement (must be within 15 words) toward the explicitly named target, such as a claim or subject, in provided context. For example, \"strongly agrees with student loan forgiveness,\" or \"somewhat disagrees with a carbon tax\". In these cases, having only \"strongly agrees\" or \"somewhat disagrees\" is not enough, as they are missing targets. If there are multiple, include all of them separated by semicolons.",
        "max_tokens": 48,
        "system_prompt": "../system_prompt/sebvgcr_stance.txt"
    },
    "emotion": 
    {
        "desc": "HUMAN's emotions with intensity (must be within 15 words) toward an explicitly named target. For example, \"Moderate heartbreak for the wildfire victims; Mild irritation about government's actions\". In this case, having only \"mild irritation,\" or \"moderate heartbreak\" are not sufficient, as the answer must express all three aspects: the emotion, the degree of emotion, and the target. If there are multiple, include all of them separated by semicolons.",
        "max_tokens": 48,
        "system_prompt": "../system_prompt/sebvgcr_emotion.txt"
    },
    "belief": 
    {
        "desc": "HUMAN's belief (must be within 15 words), namely a foundational assumption about how people, relationships, or the world fundamentally operate. Beliefs should reflect underlying mental models, not surface-level observations. Prefer beliefs that would explain multiple behaviors over beliefs that describe a single situation. Ask: \"What deeper assumption about human nature or the world would lead someone to say/do this?\" For example, \"people don't change unless they're forced to,\" \"loyalty is earned, not owed,\" \"conflict avoidance creates bigger problems later,\". Not beliefs: Practical advice, strategies, or statements about what should happen. Belief is not specific to a target or event, it should be a general statement about how HUMAN views the world.",
        "max_tokens": 48,
        "system_prompt": "../system_prompt/sebvgcr_belief.txt"
    },

```
where max_tokens determines the token limit for the model's generation for that hierarchy.

## Training

### Basic Training
```bash
bash recipe/usim/train_grpo.sh \
    <gpu_list> <dataset_name> <config_key> \
    [push_to_git] [commit_message] [resume_path] [model_type]
```

**Required Arguments:**
- `gpu_list`: Comma-separated GPU IDs (e.g., `"0,1,2,3,4,5,6,7"`)
- `dataset_name`: One of: `reddit|medium|youtube|amazon|wildchat|enron`
- `config_key`: Training configuration (see below)

**Optional Arguments:**
- `push_to_git`: `yes|no` (default: `no`)
- `commit_message`: Git commit message
- `resume_path`: Checkpoint path to resume from
- `model_type`: Model variant for `eval_only` mode (default: `base`)

### Training Configurations

#### `response_only`
Train model to generate only the final response (regular GRPO w/ no hierarchical structure):
```bash
bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6,7" reddit response_only no
```

#### `separate_system_prompt`
Train model on each state dimension (HUMANLM):
```bash
bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6" medium separate_system_prompt no
```

### Evaluation
```bash
# Evaluate base model
bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6,7" reddit eval_only no "" "" base

# Evaluate SFT checkpoint with thinking
bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6,7" wildchat eval_only no "" \
    "../outputs/sft_wildchat_english_thinking_think_r_100p/global_step_366" sft-think

# Evaluate HumanLM trained model
bash recipe/usim/train_grpo.sh "4,5,6,7" reddit eval_only no "" \
    "../outputs/rl_thinking_sep_sys_prompt_no_dropout_no_synthesis_reddit/global_step_550" \
    humanlm-sep-prompt-think
```

## Custom Components

### `create_any_dataset.py`
Processes raw conversation datasets into VERL format with hierarchical structure.

**Input:** Raw conversation data from HuggingFace  
**Output:** Processed parquet files with prompts, hierarchical responses, user profiles, and metadata for training

### `reward.py`
Custom reward manager (`UsimRewardManager`) implementing HumanLM's core training logic:
- **Hierarchical parsing**: Extracts latent states from XML-tagged responses (strict or flexible modes)
- **State alignment scoring**: LLM judge evaluates alignment of each latent state with ground truth
- **Synthesized response scoring**: Evaluates complete synthesized responses
- **Batched evaluation**: Compares rollouts within batches for more precise reward signals

## Output Structure

Checkpoints are saved to:
```
/$USER/humanlm-checkpoints/$USER/outputs/$EXP_NAME/
├── global_step_{N}/
│   ├── actor/
│   │   ├── model weights
│   │   ├── data.pt (dataloader state)
│   │   └── hierarchy_map.pt (best latent state prefixes)
│   └── critic/
│       └── model weights
└── latest_checkpointed_iteration.txt
```

The `hierarchy_map.pt` contains the highest-scoring latent states for each hierarchy level, used as prefixes for generating subsequent levels.

## Configuration Examples

See `train_grpo.sh` for full configuration options. Key parameters:
```bash
# Model settings
BASE_MODEL_TYPE=qwen  # or userlm
ENABLE_THINKING=True  # Include <think> tags for reasoning traces

# Training settings
TRAIN_EPOCHS=1
SAVE_FREQ=25  # Save every N steps
MAX_GEN_LENGTH=1024  # Increased for thinking traces

# Hierarchy settings
ENABLE_HIERARCHY=True
SEPARATE_GENERATION=True  # Generate each level separately
USE_DIFF_H_SYS_PROMPTS=True  # Different prompts per level
HIERARCHY_CONFIG='recipe/usim/hierarchy_config/sebvgcr.json'
```
