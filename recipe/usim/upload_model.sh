# ./recipe/usim/upload_model.sh  //llm_twin/outputs/rl_gpt5_judge_train_disable_thinking_sep_sys_prompt_persona_reddit/global_step_250 humanlm_reddit_best --private

# ./recipe/usim/upload_model.sh  //llm_twin/outputs/rl_gpt5_judge_train_enable_thinking_sep_sys_prompt_persona_reddit/global_step_250 humanlm_reddit_best_think --private

# ./recipe/usim/upload_model.sh  //llm_twin/outputs/rl_response_only_gpt5_judge_train_NO_PERSONA_reddit/global_step_50 grpo_ablation_reddit --private


#!/usr/bin/env bash
################################################################################
# Merge a VERL (FSDP) checkpoint and upload it to Hugging Face Hub
#
# Usage:
#   ./upload_model.sh <local_dir> <repo_name> [options]
#
# Examples:
#   # Upload RL checkpoint with default org (hf-org)
#   ./upload_model.sh /path/to/checkpoint/global_step_250 my-model-name
#
#   # Upload SFT checkpoint
#   ./upload_model.sh /path/to/sft_checkpoint my-sft-model --train_type sft
#
#   # Upload to a different org
#   ./upload_model.sh /path/to/checkpoint my-model --org my-organization
#
#   # Upload as private repo
#   ./upload_model.sh /path/to/checkpoint my-model --private
#
#   # Dry run to see what would happen
#   ./upload_model.sh /path/to/checkpoint my-model --dry_run
#
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPLOAD_SCRIPT="${SCRIPT_DIR}/upload_model.py"

# Check if upload script exists
if [[ ! -f "$UPLOAD_SCRIPT" ]]; then
    echo "Error: upload_model.py not found at: $UPLOAD_SCRIPT" >&2
    exit 1
fi

# Print usage if not enough arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <local_dir> <repo_name> [options]"
    echo ""
    echo "Required arguments:"
    echo "  local_dir    Path to the VERL checkpoint directory"
    echo "  repo_name    Name of the HF repo (without org prefix)"
    echo ""
    echo "Optional arguments:"
    echo "  --org <name>           HF organization (default: hf-org)"
    echo "  --train_type <type>    rl, sft, or pretrain (default: rl)"
    echo "  --target_root <path>   Directory for merged checkpoints"
    echo "  --private              Make HF repo private"
    echo "  --commit_message <msg> Custom commit message"
    echo "  --keep_merged          Keep merged checkpoint after upload"
    echo "  --dry_run              Print commands but don't execute"
    echo ""
    echo "Example:"
    echo "  $0 //llm_twin/outputs/rl_gpt5_judge_train_disable_thinking_sep_sys_prompt_persona_reddit/global_step_250 my-model-name"
    exit 1
fi

LOCAL_DIR="$1"
REPO_NAME="$2"
shift 2

# Validate local_dir exists
if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "Error: local_dir does not exist: $LOCAL_DIR" >&2
    exit 1
fi

echo "============================================================"
echo "VERL Model Upload to Hugging Face"
echo "============================================================"
echo "Local dir : $LOCAL_DIR"
echo "Repo name : $REPO_NAME"
echo "Options   : $*"
echo "============================================================"
echo ""

# Run the Python upload script
python3 "$UPLOAD_SCRIPT" \
    --local_dir "$LOCAL_DIR" \
    --repo_name "$REPO_NAME" \
    "$@"
