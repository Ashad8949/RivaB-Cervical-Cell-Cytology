# Riva B Cell Detection - One Epoch Test Run Guide

This guide explains how to run a complete training and testing cycle for one epoch to verify the pipeline.

## 1. Configuration

I have created a specific configuration file for this test run: `configs/test_run_config.yaml`.

Key settings in this config:
- **Epochs**: 1
- **Experiment Name**: `riva_test_run_one_epoch`
- **Data**: Uses `train_sample.csv` (small subset) for quick verification.
- **Batch Size**: 1 (safe for all GPUs/CPUs)
- **Image Size**: 512 (faster than default 1024)

## 2. Environment Setup

## 2. Environment Setup

Before running any commands, make sure to activate the local Conda environment:

```bash
conda activate ./.conda
# or if that doesn't work:
# .\.conda\Scripts\activate
pip install -r requirements.txt
```

## 3. Training

To start the one-epoch training, run the following command in your terminal:

```bash
python main.py train --config configs/test_run_config.yaml
```

This will:
1.  Load the model and data.
2.  Train for 1 epoch.
3.  Save checkpoints to `experiments/riva_test_run_one_epoch/checkpoints/`.

## 4. Testing / Inference

Once training is complete, you can run inference on the test set using the saved checkpoint.

Run this command:

```bash
python main.py submit --checkpoint experiments/riva_test_run_one_epoch/checkpoints/best.pth --test-dir riva-partb-dataset/images/test --output submission_test.csv
```

This will:
1.  Load the best model from the training run.
2.  Run inference on images in `riva-partb-dataset/images/test`.
3.  Save the results to `submission_test.csv`.

## 5. Verification

After the commands finish:
1.  **Check Training Logs**: Look at `experiments/riva_test_run_one_epoch/logs/` or the console output to ensure loss decreased.
2.  **Check Output CSV**: Open `submission_test.csv` to see the predicted bounding boxes.

## Notes

- If you want to run on the **full dataset**, edit `configs/test_run_config.yaml` and change `train_csv` to `"./riva-partb-dataset/annotations/train.csv"`. Note that this will take significantly longer.
