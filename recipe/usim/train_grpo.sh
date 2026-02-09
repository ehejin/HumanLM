# sleep for 1 h
################################################################################
# GRPO Training Script
################################################################################
# Usage: sh recipe/usim/train_grpo.sh <gpu_list> <dataset_name> <config_key> \
#                                      [push_to_git] [commit_message] [resume_path] [model_type]
#
# Arguments:
#   gpu_list      : Comma-separated GPU IDs (e.g., "0,1,2,3")
#   dataset_name  : reddit|medium|youtube|amazon|wildchat|enron
#   config_key    : response_only|separate_reward|separate_system_prompt|eval_only
#   push_to_git   : yes|no (optional, default: no)
#   commit_message: Git commit message (optional)
#   resume_path   : Path to checkpoint to resume from (optional)
#   model_type    : Model variant for eval_only mode (optional, default: base)
#                   Options: base|base-think|userlm|sft|sft-think|rl-ablation|
#                           humanlm-sep-prompt|humanlm-sep-reward
#
# Examples:
#   bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6,7" reddit eval_only no "" "" base
#   bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6,7" wildchat eval_only no "" "../outputs/sft_wildchat_english_thinking_think_r_100p/global_step_366" sft_think
#   bash recipe/usim/train_grpo.sh "4,5,6,7" reddit eval_only no "" "" base-think
#   bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6,7" reddit eval_only no "" "../outputs/rl_thinking_sep_sys_prompt_no_dropout_no_synthesis_reddit/global_step_550" humanlm-sep-prompt-think
#   bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6,7" amazon eval_only no 
#   bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6" medium response_only no
#   bash recipe/usim/train_grpo.sh "0,1,2,3,4,5,6,7" reddit separate_system_prompt no
################################################################################

set -x
set -a
source //.env
set +a

################################################################################
# ARGUMENT PARSING
################################################################################

# Check if last argument is a model type (only used with eval_only config)
MODEL_TYPE="base"
if [[ $# -ge 4 ]]; then
  last_arg="${@: -1}"
  valid_model_types="base|base-think|userlm|sft|response_only_think|sft-think|rl-ablation|rl-ablation-think|humanlm-sep-prompt|humanlm-sep-prompt-synthesis|humanlm-sep-prompt-synthesis-instruct|humanlm-sep-prompt-think|humanlm-sep-reward|gpt-5|sep-prompt-latent|sep-prompt-latent-think|humanlm-sep-reward-subset|generate-hierarchies-think|humanlm-sep-prompt-wildchat|base-wildchat|base-wildchat-think|humanlm-sep-prompt-wildchat-think|humanlm-sep-prompt-hetero|humanlm-sep-prompt-dropout|rl-ablation-wildchat|rl-ablation-think-wildchat|humanlm-sep-prompt-hetero-wildchat"
  if [[ "$last_arg" =~ ^($valid_model_types)$ ]]; then
    MODEL_TYPE="$last_arg"
    set -- "${@:1:$(($#-1))}"
  fi
fi

# Required arguments
GPU_LIST=${1:?"Error: GPU list required (e.g., '0,1,2,3')"}
DATASET_NAME=${2:?"Error: Dataset name required (reddit|medium|youtube|amazon|wildchat|enron)"}
CONFIG_KEY=${3:?"Error: Config key required"}

# Optional arguments
IS_FORMAL_RUN=${4:-no}
COMMIT=${5:-"Automatic commit before training"}
RESUME_PATH=${6:-null}

# Shift processed arguments
SHIFT_NUM=3
[[ $# -ge 4 ]] && SHIFT_NUM=4
[[ $# -ge 5 ]] && SHIFT_NUM=5
[[ $# -ge 6 ]] && SHIFT_NUM=6
shift $SHIFT_NUM

################################################################################
# ENVIRONMENT SETUP
################################################################################

PROJECT_DIR="$(pwd)"
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=$GPU_LIST
export NUM_GPUS=$(echo $GPU_LIST | awk -F',' '{print NF}')
GPU_MEMORY_UTILIZATION=$(awk "BEGIN {print $NUM_GPUS * 0.1}")

################################################################################
# CONFIGURATION BY CONFIG_KEY
################################################################################

VERL_PATH="./"
INTERLEAVE=False
ADDITIONAL_GENERATION_PROMPT=''

# Common training settings
STATES_IN_THINK=False
ITEM_DROPOUT_PROB=0.0
DROPOUT_PROB=0.0
EVAL_ONLY=False
ENABLE_HETERO_THINK=False
STRICT_FORMAT=True
VAL_DATA_FILE=val
VAL_BEFORE_TRAIN=False
VAL_METRICS='{response:{state_reward:{weight:1.0,kwargs:{model:"anthropic/claude-haiku-4-5",temperature:0}}}}'
TRAIN_EPOCHS=1
NO_REPEAT_N_GRAM_SIZE=0
EVAL_LLM_API=False
MAX_GEN_LENGTH=512
SAVE_FREQ=25
NUM_TRAIN_EXAMPLES=30
GENERATE_HIERARCHIES_ONLY_VAL=False
DROPOUT_IN_VAL=False
TRAIN_METRICS_OVERRIDE='+reward_model.reward_kwargs.train_metrics.common_kwargs={model:"openai/gpt-5-mini"}'
WILD_CHAT_FILTER_WITH_NEW_SYS_PROMPT=False
case $CONFIG_KEY in
  response_only)
    CONFIG=r
    SAVE_FREQ=50
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=False
    HIERARCHY_CONFIG='recipe/usim/hierarchy_config/r.json'
    IDENTIFIER=rl_response_only_gpt5_judge_train
    SEPARATE_GENERATION=False
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=False
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=True
    ;;

  response_only_wildchat)
    CONFIG=r
    SAVE_FREQ=50
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=False
    SUBSET_SYS_PROMPT="recipe/usim/system_prompt/wildchat_stronger/sebvgcr_response_wildchat.txt"
    HIERARCHY_CONFIG="recipe/usim/hierarchy_config/r_wildchat_stronger.json"
    IDENTIFIER=rl_response_only_gpt5_judge_train_stronger
    SEPARATE_GENERATION=False
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=False
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=True
    ;;

  response_only_think)
    CONFIG=r
    SAVE_FREQ=50
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=True
    HIERARCHY_CONFIG='recipe/usim/hierarchy_config/r.json'
    IDENTIFIER=rl_response_only_think_gpt5_judge_train
    SEPARATE_GENERATION=False
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=False
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=True
    ;;
  
  response_only_think_wildchat)
    USE_DIFF_H_SYS_PROMPTS=False
    CONFIG=r
    SAVE_FREQ=50
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=True
    SUBSET_SYS_PROMPT="recipe/usim/system_prompt/wildchat_stronger/sebvgcr_response_wildchat.txt"
    HIERARCHY_CONFIG="recipe/usim/hierarchy_config/r_wildchat_stronger.json"
    IDENTIFIER=rl_response_only_think_gpt5_judge_train_stronger_new
    SEPARATE_GENERATION=False
    SEPARATE_REWARDS=False
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=True
    ;;
  
  separate_generation)
    CONFIG=sebvgcr
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=False
    HIERARCHY_CONFIG='recipe/usim/hierarchy_config/sebvgcr.json'
    IDENTIFIER=humanlm_batched_gpt5_dis_thinking_sep_gen
    SEPARATE_GENERATION=True
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=False
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=True
    ;;

  separate_system_prompt_hetero)
    CONFIG=sebvgcr
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=True
    ITEM_DROPOUT_PROB=0.0
    DROPOUT_PROB=0.0
    IDENTIFIER=rl_thinking_sep_sys_prompt_no_dropout_hetero
    ENABLE_HETERO_THINK=True
    HIERARCHY_CONFIG='recipe/usim/hierarchy_config/sebvgcr.json'
    MAX_GEN_LENGTH=1024
    SEPARATE_GENERATION=True
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=True
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=False
    ;;
  
  separate_system_prompt_hetero_wildchat)
    CONFIG=sebvgcr
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=True
    ITEM_DROPOUT_PROB=0.0
    DROPOUT_PROB=0.0
    HIERARCHY_CONFIG='recipe/usim/hierarchy_config/sebvgcr_wildchat_stronger.json'
    IDENTIFIER=rl_thinking_sep_sys_prompt_no_dropout_hetero
    ENABLE_HETERO_THINK=True
    MAX_GEN_LENGTH=1024
    SEPARATE_GENERATION=True
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=True
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=False
    ;;

  separate_system_prompt)
    CONFIG=sebvgcr
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=False
    MAX_GEN_LENGTH=512
    HIERARCHY_CONFIG='recipe/usim/hierarchy_config/sebvgcr.json'
    IDENTIFIER=humanlm_512_disable_thinking_sep_sys_prompt
    ITEM_DROPOUT_PROB=0.0
    DROPOUT_PROB=0.0
    DROPOUT_IN_VAL=False
    SEPARATE_GENERATION=True
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=True
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=False
    ;;
  
  separate_system_prompt_wildchat)
    CONFIG=sebvgcr
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=False
    MAX_GEN_LENGTH=512
    HIERARCHY_CONFIG='recipe/usim/hierarchy_config/sebvgcr_wildchat_stronger.json'
    IDENTIFIER=humanlm_512_disable_thinking_sep_sys_prompt2
    ITEM_DROPOUT_PROB=0.0
    DROPOUT_PROB=0.0
    DROPOUT_IN_VAL=False
    SEPARATE_GENERATION=True
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=True
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=False
    ;;

  separate_reward_subset)
    SUBSET_SYS_PROMPT="./recipe/usim/system_prompt/ser.txt"
    CONFIG=sebvgcr
    BASE_MODEL_TYPE=qwen
    ENABLE_THINKING=False
    HIERARCHY_CONFIG='recipe/usim/hierarchy_config/ser.json'
    IDENTIFIER=humanlm_batched_gpt5_dis_thinking_ser_sep_rewards
    SEPARATE_GENERATION=False
    SEPARATE_REWARDS=True
    USE_DIFF_H_SYS_PROMPTS=False
    CONCATENATE_HIERARCHY_BATCHES=False
    ENABLE_HIERARCHY=True
    ;;

  eval_only)
    export VLLM_USE_V1=0  
    NO_REPEAT_N_GRAM_SIZE=4
    
    # MODEL_TYPE determines the specific configuration variant 
    case "$MODEL_TYPE" in
      base)                     CONFIG=r               ENABLE_THINKING=False     BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False ;;
      base-think)               CONFIG=r               ENABLE_THINKING=True      BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False ;;
      base-wildchat)            CONFIG=r_wildchat_stronger   DATA_CONFIG=r   ENABLE_THINKING=False    BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_wildchat_template.jinja";;
      base-wildchat-think)            CONFIG=r_wildchat_stronger   DATA_CONFIG=r   ENABLE_THINKING=True    BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_wildchat_template.jinja";;
      gpt-5)                    CONFIG=r               ENABLE_THINKING=False     BASE_MODEL_TYPE=qwen   EVAL_LLM_API=True  ;;
      sft)                      CONFIG=r_no_tag        ENABLE_THINKING=False     BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False ;;
      sft-think)                CONFIG=think_r         ENABLE_THINKING=True      BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False;;
      userlm)                   CONFIG=r_no_tag_userlm    DATA_CONFIG=r_no_tag    ENABLE_THINKING=False     BASE_MODEL_TYPE=userlm EVAL_LLM_API=False  USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True;;
      userlm-wildchat)                   CONFIG=r_no_tag_userlm_wildchat    DATA_CONFIG=r_no_tag    ENABLE_THINKING=False     BASE_MODEL_TYPE=userlm EVAL_LLM_API=False  USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True;;
      rl-ablation)              CONFIG=r               ENABLE_THINKING=False     BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False;;
      rl-ablation-think)        CONFIG=r               ENABLE_THINKING=True      BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False;;
      rl-ablation-wildchat)    CONFIG=r_wildchat_stronger   DATA_CONFIG=r   ENABLE_THINKING=False    BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_wildchat_template.jinja";;
      rl-ablation-think-wildchat)    CONFIG=r_wildchat_stronger   DATA_CONFIG=r   ENABLE_THINKING=True    BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_wildchat_template.jinja";;
      humanlm-sep-prompt)         CONFIG=r               ENABLE_THINKING=False     BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False;;
      humanlm-sep-prompt-dropout) CONFIG=r               ENABLE_THINKING=False     BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False   ITEM_DROPOUT_PROB=0.8   DROPOUT_IN_VAL=True ;;
      humanlm-sep-prompt-think)   CONFIG=r              ENABLE_THINKING=True    BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False;;
      generate-hierarchies-think) CONFIG=r              ENABLE_THINKING=True    BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False   USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True;;
      humanlm-sep-reward)         CONFIG=sebvgcr         ENABLE_THINKING=False     BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False;;
      humanlm-sep-prompt-hetero)   CONFIG=r    ENABLE_THINKING=True   ENABLE_HETERO_THINK=True   BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False  MAX_GEN_LENGTH=1024;;
      humanlm-sep-prompt-hetero-wildchat)  CONFIG=r_wildchat_stronger   ENABLE_THINKING=True ENABLE_HETERO_THINK=True   BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False MAX_GEN_LENGTH=1024  USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_wildchat_template.jinja";;
      humanlm-sep-prompt-synthesis-instruct)   CONFIG=r_synthesis_instruct  DATA_CONFIG=r  ENABLE_THINKING=True  BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False  USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True;;
      humanlm-sep-prompt-wildchat)   CONFIG=sebvgcr_wildchat_stronger   DATA_CONFIG=r   ENABLE_THINKING=False    BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_wildchat_template.jinja";;
      humanlm-sep-prompt-wildchat-think) CONFIG=sebvgcr_wildchat_stronger   DATA_CONFIG=r   ENABLE_THINKING=True    BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_wildchat_template.jinja";;
      humanlm-sep-reward-subset)       CONFIG=sebvgcr         ENABLE_THINKING=False     BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False  SUBSET_SYS_PROMPT="./recipe/usim/system_prompt/ser.txt";;
      humanlm-sep-prompt-synthesis)     CONFIG=r_synthesis  DATA_CONFIG=r  ENABLE_THINKING=False  BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False  USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True;;
      humanlm-sep-prompt-latent)        CONFIG=r_latent_think  DATA_CONFIG=r  ENABLE_THINKING=False  BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False  USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True;;
      humanlm-sep-prompt-latent-think)  CONFIG=r_latent_think  DATA_CONFIG=r  ENABLE_THINKING=True   BASE_MODEL_TYPE=qwen   EVAL_LLM_API=False  USE_DIFF_H_SYS_PROMPTS_OVERRIDE=True;;
      *)
        echo "Error: Unknown MODEL_TYPE: $MODEL_TYPE" >&2
        exit 1
        ;;
    esac
    VAL_SIZE=2000
    EVAL_ONLY=True
    STRICT_FORMAT=False
    HIERARCHY_CONFIG="recipe/usim/hierarchy_config/$CONFIG.json"
    IDENTIFIER=eval_qwen3_$MODEL_TYPE
    SEPARATE_GENERATION=False
    SEPARATE_REWARDS=False
    USE_DIFF_H_SYS_PROMPTS=False
    # Override USE_DIFF_H_SYS_PROMPTS if MODEL_TYPE requires it (e.g., sep-prompt-latent*)
    if [[ -n "$USE_DIFF_H_SYS_PROMPTS_OVERRIDE" ]]; then
      USE_DIFF_H_SYS_PROMPTS=True
      SEPARATE_GENERATION=True
    fi
    CONCATENATE_HIERARCHY_BATCHES=False
    VAL_DATA_FILE=test
    VAL_BEFORE_TRAIN=True
    VAL_METRICS='{response:{state_reward_on_response:{weight:1.0,kwargs:{model:"anthropic/claude-haiku-4-5",temperature:0,config_path:"../digital-human-lm/verl/recipe/usim/hierarchy_config/sebvgc.json"}},state_reward:{weight:1.0,kwargs:{model:"anthropic/claude-haiku-4-5",temperature:0}}}}'
    ENABLE_HIERARCHY=True
    set -- "+reward_model.reward_kwargs.debug_logs=true" "$@"
    TRAIN_EPOCHS=0
    ;;

  *)
    echo "Error: Invalid config key '$CONFIG_KEY'"
    exit 1
    ;;
esac




# Assert: USE_DIFF_H_SYS_PROMPTS=True => SEPARATE_GENERATION=True
if [[ "$USE_DIFF_H_SYS_PROMPTS" == "True" && "$SEPARATE_GENERATION" != "True" ]]; then
  echo "Error: USE_DIFF_H_SYS_PROMPTS=True requires SEPARATE_GENERATION=True." >&2
  echo "       Got: USE_DIFF_H_SYS_PROMPTS=$USE_DIFF_H_SYS_PROMPTS, SEPARATE_GENERATION=$SEPARATE_GENERATION" >&2
  echo "       Context: CONFIG_KEY=$CONFIG_KEY, MODEL_TYPE=$MODEL_TYPE" >&2
  exit 1
fi

################################################################################
# MODEL CONFIGURATION
################################################################################

case $BASE_MODEL_TYPE in
  qwen)   MODEL_PATH=//llm_twin/models/Qwen3-8B; CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_multi_role_template_think.jinja" ;;
  userlm) MODEL_PATH=//llm_twin/models/UserLM-8B; CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/userlm_multi_role_chat_template.jinja" ;;
  *)
    echo "Error: Invalid model type '$BASE_MODEL_TYPE'. Use: qwen|userlm"
    exit 1
    ;;
esac

if [[ "$DATASET_NAME" == "wildchat" && "$BASE_MODEL_TYPE" == "qwen" ]]; then
  CHAT_TEMPLATE="$VERL_PATH/recipe/usim/chat_templates/qwen3_wildchat_template.jinja"
  VAL_METRICS='{response:{state_reward_on_response_wildchat2:{weight:1.0,kwargs:{model:"anthropic/claude-haiku-4-5",temperature:0,config_path:"./recipe/usim/hierarchy_config/sebvgc_wildchat_stronger.json"}},state_reward_wildchat2:{weight:1.0,kwargs:{model:"anthropic/claude-haiku-4-5",temperature:0}}}}'
  TRAIN_METRICS_OVERRIDE='+reward_model.reward_kwargs.train_metrics={common_kwargs: {model: "openai/gpt-5-mini"}, response: {state_reward_wildchat2: {weight: 1.0, kwargs: {}}}}'
  WILD_CHAT_FILTER_WITH_NEW_SYS_PROMPT=True
fi

# if enable thinking and max_gen_length is less than 1024, set to 1024
if [[ "$ENABLE_THINKING" == "True" && "$MAX_GEN_LENGTH" -lt 1024 ]]; then
  MAX_GEN_LENGTH=1024
fi

################################################################################
# Merge SFT checkpoints for eval_only runs
################################################################################
MERGE_SCRIPT="${VERL_PATH}/recipe/usim/fsdp_merge.py"
ID=$(ls /lfs | grep '^ampere' | sed 's/ampere//')      
MERGED_TARGET_ROOT="/lfs/ampere${ID}/0/${USER}"

if [[ "$CONFIG_KEY" == "eval_only" && ( "$MODEL_TYPE" == "sft" || "$MODEL_TYPE" == "sft-think" ) ]]; then
  if [[ "$RESUME_PATH" == "null" || -z "$RESUME_PATH" ]]; then
    echo "Error: For MODEL_TYPE=$MODEL_TYPE, you must pass a resume_path pointing to SFT checkpoint." >&2
    exit 1
  fi

  if [[ ! -f "$MERGE_SCRIPT" ]]; then
    echo "Error: merge script not found at: $MERGE_SCRIPT" >&2
    exit 1
  fi

  echo "[merge] Merging SFT checkpoint for eval: RESUME_PATH=$RESUME_PATH"
  # merge_model.py prints logs then prints the merged dir, we use last line as path
  MERGED_DIR="$(python3 "$MERGE_SCRIPT" \
      --local_dir "$RESUME_PATH" \
      --label "$EXP_NAME" \
      --target_root "$MERGED_TARGET_ROOT" \
    | tail -n 1)"

  echo "[merge] Using merged model dir as MODEL_PATH: $MERGED_DIR"
  MODEL_PATH=$MERGED_DIR
  RESUME_PATH=null
fi

################################################################################
# Set Resume Mode
################################################################################

RESUME_MODE="auto"
if [[ -n "$RESUME_PATH" && "$RESUME_PATH" != "null" ]]; then
  RESUME_MODE="resume_path"
fi

##################################################################
#                     DATASET CONFIGURATION                      #
##################################################################

DATASET_DIR=//llm_twin/processed_data
FILTER_OVERLONG_PROMPTS=True

# Use DATA_CONFIG for data path if set, otherwise fall back to CONFIG
# This allows using different hierarchy configs (for system prompts) with existing data
DATA_CONFIG_FOR_PATH="${DATA_CONFIG:-$CONFIG}"

BATCH_SIZE=$((32 / NUM_GPUS * NUM_GPUS))
case $DATASET_NAME in
  reddit)   FILTER_OVERLONG_PROMPTS=False; MAX_LENGTH=5120; BATCH_SIZE=$BATCH_SIZE; DATA_PATH="$DATASET_DIR/reddit_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/50p" ;;
  medium)   MAX_LENGTH=7168; BATCH_SIZE=$BATCH_SIZE; DATA_PATH="$DATASET_DIR/medium_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/25p" ;;
  youtube)  MAX_LENGTH=5120; BATCH_SIZE=$BATCH_SIZE; DATA_PATH="$DATASET_DIR/youtube_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/5p" ;;
  amazon)   MAX_LENGTH=7168; BATCH_SIZE=$BATCH_SIZE; DATA_PATH="$DATASET_DIR/amazon_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/100p" ;;
  wildchat) MAX_LENGTH=7168; BATCH_SIZE=$BATCH_SIZE; DATA_PATH="$DATASET_DIR/wildchat_english_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/100p" ;;
  enron)    MAX_LENGTH=5120; BATCH_SIZE=$BATCH_SIZE; DATA_PATH="$DATASET_DIR/enron_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/100p" ;;
  *)
    echo "Error: Invalid dataset '$DATASET_NAME'"
    echo "Valid options: reddit|medium|youtube|amazon|wildchat|enron"
    exit 1
    ;;
esac

if [[ "$DATASET_NAME" == "reddit" ]]; then
  if [[ "$EVAL_ONLY" == "True" ]]; then
    VAL_SIZE=5000
  else
    VAL_SIZE=1500
  fi
else
  if [[ "$EVAL_ONLY" == "True" ]]; then
    VAL_SIZE=2000
  else
    VAL_SIZE=500
  fi
fi


EXP_NAME="${IDENTIFIER}_${DATASET_NAME}"

################################################################################
# DISPLAY CONFIGURATION SUMMARY
################################################################################

cat <<EOF
================================================================================
                          Configuration Summary
================================================================================
GPUs:                 $GPU_LIST ($NUM_GPUS GPUs)
GPU Memory Util:      $GPU_MEMORY_UTILIZATION
Model Type:           $MODEL_TYPE
Model Path:           $MODEL_PATH
Dataset:              $DATASET_NAME
Config Key:           $CONFIG_KEY
Max Length:           $MAX_LENGTH
Batch Size:           $BATCH_SIZE
Data Path:            $DATA_PATH
Experiment Name:      $EXP_NAME
Commit Message:       $COMMIT
Resume Path:          $RESUME_PATH
Push to Git:          $IS_FORMAL_RUN
Training Epochs:      $TRAIN_EPOCHS
================================================================================
EOF

################################################################################
# GIT WORKFLOW (if formal run)
################################################################################

if [[ "$IS_FORMAL_RUN" == "yes" ]]; then
    export HOME=/dfs/user/$USER
    GIT_SSH="ssh -i /dfs/user/$USER/.ssh/id_rsa -o IdentitiesOnly=yes -o UserKnownHostsFile=/dfs/user/$USER/.ssh/known_hosts -o StrictHostKeyChecking=accept-new"
    BRANCH_NAME="${USER}_${IDENTIFIER}"
    
    echo "Committing and pushing to main branch..."
    git add .
    git commit -m "$COMMIT"
    GIT_SSH_COMMAND="$GIT_SSH" git push origin 001_finalize_train
    
    # Create or checkout feature branch
    if git show-ref --verify --quiet refs/heads/$BRANCH_NAME; then
        echo "Checking out existing branch: $BRANCH_NAME"
        git checkout $BRANCH_NAME
        GIT_SSH_COMMAND="$GIT_SSH" git pull origin 001_finalize_train
    else
        echo "Creating new branch: $BRANCH_NAME"
        git checkout -b $BRANCH_NAME
    fi
    
    GIT_SSH_COMMAND="$GIT_SSH" git push origin $BRANCH_NAME
    git checkout 001_finalize_train
    
    echo "Git workflow completed successfully"
fi

################ FIXED #############################
OUTPUT_DIR="/$USER/humanlm-checkpoints/$USER/outputs/$EXP_NAME"
CACHE_DIR="/verl_cache"
export NEW_HF_CACHE=//llm_twin/hf-cache/$USER
RUNTIME_CACHE="/runtime_cache/$USER/"

export XDG_CACHE_HOME="$RUNTIME_CACHE/xdg"        
export VLLM_CACHE_ROOT="$RUNTIME_CACHE/vllm"      
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_CACHE/torchinductor"
export TRITON_CACHE_DIR="$RUNTIME_CACHE/triton"
export WANDB_ENTITY=dsp-team
export HF_HOME="$NEW_HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$NEW_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$NEW_HF_CACHE/hub"
export HF_DATASETS_CACHE="$NEW_HF_CACHE/datasets"
export VERL_CACHE_DIR="$NEW_HF_CACHE/verl-cache"

export RAY_raylet_client_num_connect_attempts=20
export RAY_raylet_client_connect_timeout_milliseconds=500000


python3 -m verl.trainer.main_hierarchy \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    '+trainer.load_hierarchy_map=False' \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=usim \
    custom_reward_function.path="$VERL_PATH/recipe/usim/reward.py" \
    custom_reward_function.name="compute_reward" \
    +reward_model.reward_kwargs.enable_hierarchy=$ENABLE_HIERARCHY \
    '+reward_model.reward_kwargs.fetch_global_best_hierarchy=True' \
    +reward_model.reward_kwargs.separate_generation=$SEPARATE_GENERATION \
    +reward_model.reward_kwargs.separate_rewards=$SEPARATE_REWARDS \
    +reward_model.reward_kwargs.eval_llm_api=$EVAL_LLM_API \
    +reward_model.reward_kwargs.enable_thinking=$ENABLE_THINKING \
    +reward_model.reward_kwargs.eval_push_to_hub="hf-org/$EXP_NAME-split_$VAL_DATA_FILE" \
    +reward_model.reward_kwargs.states_in_think=$STATES_IN_THINK \
    +reward_model.reward_kwargs.hierarchy_config=$HIERARCHY_CONFIG \
    +reward_model.reward_kwargs.val_hierarchy_config=$VAL_HIERARCHY_CONFIG \
    +reward_model.reward_kwargs.strict_format=$STRICT_FORMAT \
    "$TRAIN_METRICS_OVERRIDE" \
    +reward_model.reward_kwargs.val_metrics=$VAL_METRICS \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/$VAL_DATA_FILE.parquet \
    +data.cache_dir=$CACHE_DIR \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=256 \
    +data.kwargs.multirole_chat_template_path="$CHAT_TEMPLATE" \
    '+data.seed=0' \
    data.max_response_length=$MAX_GEN_LENGTH \
    data.max_prompt_length=$MAX_LENGTH \
    data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS \
    data.truncation='error' \
    data.filter_overlong_prompts_workers=128 \
    +data.hierarchy_config_path=$HIERARCHY_CONFIG \
    +data.enable_hetero_think=$ENABLE_HETERO_THINK \
    +data.augment_with_hierarchies=$USE_DIFF_H_SYS_PROMPTS \
    +data.separate_rewards=$SEPARATE_REWARDS \
    +data.separate_generation=$SEPARATE_GENERATION \
    +data.val_size=$VAL_SIZE \
    +data.eval_only=$EVAL_ONLY \
    +data.is_wildchat=$WILD_CHAT_FILTER_WITH_NEW_SYS_PROMPT \
    +data.generate_hierarchies=$GENERATE_HIERARCHIES_ONLY_VAL \
    +data.dataset=$DATASET_NAME \
    +data.new_sys_prompt=$SUBSET_SYS_PROMPT \
    +data.field_dropout_prob=$DROPOUT_PROB \
    +data.item_dropout_prob=$ITEM_DROPOUT_PROB \
    +reward_model.reward_kwargs.additional_generation_prompt=$ADDITIONAL_GENERATION_PROMPT \
    +data.additional_generation_prompt=$ADDITIONAL_GENERATION_PROMPT \
    +data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    +actor_rollout_ref.kwargs.custom_chat_template="$CHAT_TEMPLATE" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    +actor_rollout_ref.rollout.no_repeat_ngram_size=$NO_REPEAT_N_GRAM_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    '+actor_rollout_ref.rollout.stop="</response>"' \
    +trainer.do_distribution_eval=False \
    trainer.resume_mode=$RESUME_MODE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='usim' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    +trainer.interleave_ground_truth=$INTERLEAVE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$SAVE_FREQ \
    +trainer.num_train_examples=$NUM_TRAIN_EXAMPLES \
    +trainer.concatenate_hierarchy_batches=$CONCATENATE_HIERARCHY_BATCHES \
    +trainer.update_hierarchy_loader_freq=3 \
    +trainer.save_hierarchy_map_freq=50 \
    trainer.default_hdfs_dir=null \
    trainer.log_val_generations=20 \
    trainer.resume_from_path=$RESUME_PATH \
    trainer.total_epochs=$TRAIN_EPOCHS $@