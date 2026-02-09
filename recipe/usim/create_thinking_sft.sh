#!/bin/bash
# Script to generate thinking SFT data
# This generates synthetic thinking traces using LLM calls
#
# Usage:
#   bash recipe/usim/create_thinking_sft.sh <dataset|all> [thinking_model] [thinking_batch_size]
#
# Examples:
#   bash recipe/usim/create_thinking_sft.sh youtube
#   bash recipe/usim/create_thinking_sft.sh reddit
#   bash recipe/usim/create_thinking_sft.sh youtube gpt-4o-mini 20
#   bash recipe/usim/create_thinking_sft.sh all

set -e
set -x

TARGET=${1:?"Error: dataset required (medium|youtube|reddit|amazon|enron|wildchat|all)"}
THINKING_MODEL=${2:-"gpt-4o-mini"}
THINKING_BATCH_SIZE=${3:-200}

DATASET_ROOT="//llm_twin/processed_data_dup"
PROMPT_DIR="./recipe/usim/system_prompt/"
# For thinking_sft, the response is wrapped as:
#   <think>...</think><response>...</response>
# so the hierarchy config must define BOTH tags (think + response).
HIERARCHY_CONFIG="./recipe/usim/hierarchy_config/think_r.json"

DATASETS=()
case "$TARGET" in
  all)     DATASETS=(medium youtube reddit amazon enron wildchat) ;;
  medium)  DATASETS=(medium) ;;
  youtube) DATASETS=(youtube) ;;
  reddit)  DATASETS=(reddit) ;;
  amazon)  DATASETS=(amazon) ;;
  enron)   DATASETS=(enron) ;;
  wildchat) DATASETS=(wildchat) ;;
  *)
    echo "Error: invalid dataset '$TARGET' (use: medium|youtube|reddit|amazon|enron|wildchat|all)" >&2
    exit 1
    ;;
esac

for NAME in "${DATASETS[@]}"; do
  # Hardcoded percentages (do not override via CLI to avoid accidental expensive runs)
  case "$NAME" in
    amazon)  TP=100   ;;
    reddit)  TP=50  ;;
    medium)  TP=25  ;;
    youtube) TP=5   ;;
    enron)   TP=100 ;;
    wildchat) TP=100 ;;
    *) echo "Error: unsupported dataset '$NAME'" >&2; exit 1 ;;
  esac
  TEP=100

  # wildchat in this repo is "wildchat_english"
  DATASET_KEY="$NAME"
  if [[ "$NAME" == "wildchat" ]]; then
    DATASET_KEY="wildchat_english"
  fi

  DATASET_NAME="${DATASET_KEY}_processed_dataset_by_post_dedup"

    python -m recipe.usim.create_any_dataset \
    --dataset "$DATASET_KEY" \
    --raw_dataset_repo "hf-org/$DATASET_NAME" \
    --save_data_dir "$DATASET_ROOT/$DATASET_NAME" \
    --save_prompt_dir "$PROMPT_DIR" \
    --hierarchy_config_path "$HIERARCHY_CONFIG" \
        --sft \
        --thinking_sft \
    --thinking_model "$THINKING_MODEL" \
    --thinking_batch_size "$THINKING_BATCH_SIZE" \
    --train_subset_percentage "$TP" \
    --test_subset_percentage "$TEP"
done

