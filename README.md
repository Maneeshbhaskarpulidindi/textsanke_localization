# TextSnake Localization

This repository contains the training and inference pipeline for text localization based on the TextSnake model.

## Quick Start

The main entry point for the training and testing workflow is the `run_pipeline.sh` script. 

```bash
./run_pipeline.sh
```

This script will:
1. Train the TextSnake model on your custom dataset (`train_textsnake_mod.py`).
2. Run inference on a batch of test images and generate predicted text visualizations (`inference_custom.py`).

*Note: Before running, update `DATA_ROOT` and `TEST_IMGS` in `run_pipeline.sh` to point to your dataset and testing folders.*

## Key Scripts

- **`train_textsnake_mod.py`**: The training script. It handles data augmentation, custom dataloading, and model checkpoints.
- **`inference_custom.py`**: The inference script. It runs given test images through the trained model, performs post-processing on the geometry maps, and outputs bounding polygon visualisations.
- **`eval_textsnake.py`**: Evaluation script used for generating detection metrics like Precision, Recall, and F-measure via the DetEval protocol.
- **`util/`**: Contains utility functions for configuration (`config.py`), visualizations (`visualize.py`), data augmentation, and text center line post-processing (`detection.py`).

## Running Inference Independently

If you already have pre-trained weights (e.g., `best.pth`) and only want to perform inference and view the visualizations, you can invoke the inference script directly. Make sure you are using the correct conda environment (e.g. `conda activate textsnake`).

```bash
python inference_custom.py \
    --exp_name "custom_training_enhanced" \
    --model_path "best.pth" \
    --img_path "images_test" \
    --output_dir "newlogs/inference_test"
```

## Directory Structure
- **`images_test/`**: Folder containing sample images for inference testing.
- **`newlogs/save/`**: The default location where training checkpoints (like `best.pth`) are stored.
- **`newlogs/inference_test/`**: The default location where the inference visualizations are saved after running inference.
