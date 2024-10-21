
# Video Classification Home Assignment

This repository contains the final submission for the Workout Video Classification task. The objective of this project is to classify workout movements from short video clips, using different model architectures to analyze performance.

## Overview

This project explores different methods to solve the workout video classification problem. Over the course of experimentation, I explored multiple strategies for:
- Data preprocessing
- Model architecture

### Models Tested:
1. **ResNet18**: A pretrained model, modified by adding two fully connected layers to the pretrained ResNet18 model from `torchvision.models`.
2. **C3D**: A 3D Convolutional model.

Choice of models was influenced by the fact that temporal information may not be critical for this task, as the videos in the dataset are short (around 2 seconds).

The dataset was split into training, validation, and test sets, with the partition stored in a `.csv` file for reproducibility.

The final model was trained in the Kaggle notebook environment, utilizing a GPU P100 accelerator. The saved model is located in the path:  
`/kaggle/input/task_models/pytorch/final_model/2/final_model.pth`

## Requirements

To install missing packages:
```bash
pip install -r requirements.txt
```

## Usage
You can run the notebook in a Kaggle environment or locally. If running locally, ensure that the necessary video processing dependencies such as `imageio`, `pyav`, and `opencv-python` are installed.

If using GPU acceleration, the notebook will automatically detect and utilize it.

## Results
Final model accuracy, F1 scores, and confusion matrices are available in the notebook. The ResNet18 model with fine-tuning performed the best on the workout video classification task.
