#!/bin/bash

# Configuration
DATA_ROOT="/home/frinks/maneesh/Mar 8/text_sanke/dummy 1/dummy"
TEST_IMGS="/home/frinks/maneesh/Mar 8/textsanke_localization/images_test"
EXP_NAME="custom_training_enhanced"
EPOCHS=600

echo "=========================================================="
echo " Starting TextSnake Training Pipeline "
echo " Data Root:       $DATA_ROOT"
echo " Experiment Name: $EXP_NAME"
echo " Max Epochs:      $EPOCHS"
echo " Logs Dir:        newlogs/"
echo "=========================================================="

# 1. Run the training script.
# Utilizing the 'custom' dataset type which expects --data_root
python train_textsnake_mod.py \
    "$EXP_NAME" \
    --dataset custom \
    --data_root "$DATA_ROOT" \
    --batch_size 4 \
    --viz

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting pipeline."
    exit 1
fi

echo "Training completed successfully."
echo "Best model saved to: newlogs/save/$EXP_NAME/best.pth"

echo "=========================================================="
echo " Starting Inference Pipeline on Test Images"
echo " Test Images: $TEST_IMGS"
echo " Output Dir:  newlogs/inference_test"
echo "=========================================================="

# 2. Run inference using the best saved weights.
# We map the saved location directly to the inference script.
python inference_custom.py \
    --exp_name "$EXP_NAME" \
    --model_path "newlogs/save/best.pth" \
    --img_path "$TEST_IMGS" \
    --output_dir "newlogs/inference_test" 

echo "Inference completed."
echo "Check newlogs/inference_test for results, and newlogs/logs to view TensorBoard metrics and visuals."
