
# Video Classification Home Assignment

This repository contains the final submission for the Workout Video Classification task. The objective of this project is to classify workout movements from short video clips, using different model architectures to analyze performance.

## Table of Contents
- [Overview](#overview)
- [Helper Functions](#helper-functions)
- [Dataset Configuration](#dataset-configuration)
- [Dataset Initialization](#dataset-initialization)
- [Imbalanced vs Balanced Data](#imbalanced-vs-balanced-data)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## Overview

This project explores different methods to solve the workout video classification problem. Over the course of experimentation, I explored multiple strategies for:
- Data preprocessing
- Model architecture
- Evaluation metrics

### Models Tested:
1. **ResNet18**: A pretrained model with two fully connected layers.
2. **C3D**: A 3D Convolutional model.

The choice of these models was influenced by the fact that temporal information may not be critical for this task, as the videos in the dataset are short (around 2 seconds).

The dataset was split into training, validation, and test sets, with the partition stored in a `.csv` file for reproducibility.

The final model was trained in the Kaggle notebook environment, utilizing a GPU P100 accelerator. The saved model is located in the path:  
`/kaggle/input/task_models/pytorch/final_model/2/final_model.pth`

## Helper Functions
Custom helper functions for loading videos, data augmentation, and preprocessing.

## Dataset Configuration
The dataset consists of workout videos, where each video represents a particular exercise. The labels are extracted from the folder names.

## Dataset Initialization
- Training, Validation, and Test sets were initialized using a stratified split approach to ensure balanced classes across datasets.
- **Imbalanced vs. Balanced Data**: This notebook also compares results between training on imbalanced vs. balanced datasets using sampling strategies.

## Model Architecture

- **ResNet18**: The architecture was modified by adding two fully connected layers to the pretrained ResNet18 model from `torchvision.models`.
- **C3D**: A 3D Convolutional Neural Network for video data processing.

## Evaluation
- **Metrics**: 
  - Accuracy
  - F1 Score
  - Confusion Matrix
- Evaluation results were generated on the test set after training.

## Requirements
The following Python libraries were used:
- `numpy`
- `pandas`
- `torch`
- `torchvision`
- `opencv-python`
- `scikit-learn`
- `imageio`
- `matplotlib`

To install missing packages:
```bash
pip install -r requirements.txt
```

## Usage
You can run the notebook in a Kaggle environment or locally. If running locally, ensure that the necessary video processing dependencies such as `imageio`, `pyav`, and `opencv-python` are installed.

If using GPU acceleration, the notebook will automatically detect and utilize it.

## Results
Final model accuracy, F1 scores, and confusion matrices are available in the notebook. The ResNet18 model with fine-tuning performed the best on the workout video classification task.

## License
This project is licensed under the MIT License.
