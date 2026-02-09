CONFIG_LIST="sebvgcr"
DATA_DIR=//llm_twin/processed_data

# 
for NAME in reddit medium youtube amazon enron wildchat_english; do
    # Set percentage based on dataset 
    if [ "$NAME" = "amazon" ]; then
        TRAIN_PERCENT=100
        TEST_PERCENT=100
    elif [ "$NAME" = "reddit" ]; then
        TRAIN_PERCENT=50
        TEST_PERCENT=100
    elif [ "$NAME" = "medium" ]; then
        TRAIN_PERCENT=25
        TEST_PERCENT=50
    elif [ "$NAME" = "youtube" ]; then
        TRAIN_PERCENT=5
        TEST_PERCENT=100
    elif [ "$NAME" = "wildchat_english" ]; then
        TRAIN_PERCENT=100
        TEST_PERCENT=100
    elif [ "$NAME" = "enron" ]; then
        TRAIN_PERCENT=100
        TEST_PERCENT=100
    fi

    echo "Processing dataset: $NAME with config: $CONFIG at $TRAIN_PERCENT% train subset and $TEST_PERCENT% test subset"
    DATASET_NAME=${NAME}_processed_dataset_by_post_dedup
    
    # RL on response | for eval sft think - system prompt is think_r
    python -m recipe.usim.create_any_dataset \
        --dataset $NAME \
        --raw_dataset_repo hf-org/$DATASET_NAME \
        --save_data_dir $DATA_DIR/$DATASET_NAME \
        --save_prompt_dir ./recipe/usim/system_prompt/ \
        --hierarchy_config_path ./recipe/usim/hierarchy_config/think_r.json \
        --train_subset_percentage $TRAIN_PERCENT --test_subset_percentage $TEST_PERCENT

    # for CONFIG in $CONFIG_LIST; do
    #     # RL (config) | for HumanLM separate reward training
    #     python -m recipe.usim.create_any_dataset \
    #         --dataset $NAME \
    #         --raw_dataset_repo hf-org/$DATASET_NAME \
    #         --save_data_dir $DATA_DIR/$DATASET_NAME \
    #         --save_prompt_dir ./recipe/usim/system_prompt/ \
    #         --hierarchy_config_path ./recipe/usim/hierarchy_config/$CONFIG.json \
    #         --train_subset_percentage $TRAIN_PERCENT --test_subset_percentage $TEST_PERCENT
    # done
    
    # # RL on response (no tag) | for eval base, base-think, userlm, sft, sft-think, response-only
    # python -m recipe.usim.create_any_dataset \
    #     --dataset $NAME \
    #     --raw_dataset_repo hf-org/$DATASET_NAME \
    #     --save_data_dir $DATA_DIR/$DATASET_NAME \
    #     --save_prompt_dir ./recipe/usim/system_prompt/ \
    #     --hierarchy_config_path ./recipe/usim/hierarchy_config/r.json \
    #     --no_tag \
    #     --train_subset_percentage $TRAIN_PERCENT --test_subset_percentage $TEST_PERCENT

    # # RL on response | for train and eval humanlm - different system prompt
    # python -m recipe.usim.create_any_dataset \
    #     --dataset $NAME \
    #     --raw_dataset_repo hf-org/$DATASET_NAME \
    #     --save_data_dir $DATA_DIR/$DATASET_NAME \
    #     --save_prompt_dir ./recipe/usim/system_prompt/ \
    #     --hierarchy_config_path ./recipe/usim/hierarchy_config/r.json \
    #     --train_subset_percentage $TRAIN_PERCENT --test_subset_percentage $TEST_PERCENT

    # # SFT on response (no tag) | for train sft
    # python -m recipe.usim.create_any_dataset \
    #     --dataset $NAME \
    #     --raw_dataset_repo hf-org/$DATASET_NAME \
    #     --save_data_dir $DATA_DIR/$DATASET_NAME \
    #     --save_prompt_dir ./recipe/usim/system_prompt/ \
    #     --hierarchy_config_path ./recipe/usim/hierarchy_config/r.json \
    #     --sft \
    #     --no_tag \
    #     --train_subset_percentage $TRAIN_PERCENT --test_subset_percentage $TEST_PERCENT

    # # SFT on response | not used
    # python -m recipe.usim.create_any_dataset \
    #     --dataset $NAME \
    #     --raw_dataset_repo hf-org/$DATASET_NAME \
    #     --save_data_dir $DATA_DIR/$DATASET_NAME \
    #     --save_prompt_dir ./recipe/usim/system_prompt/ \
    #     --hierarchy_config_path ./recipe/usim/hierarchy_config/r.json \
    #     --sft \
    #     --train_subset_percentage $TRAIN_PERCENT --test_subset_percentage $TEST_PERCENT
    
done

chmod -R g+w $DATA_DIR/