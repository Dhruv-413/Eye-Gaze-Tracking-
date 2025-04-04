# Gaze Tracking System

## Introduction
This repository contains tools for capturing, processing, and analyzing gaze data. It includes scripts for data extraction, preprocessing, and model training or inference.

## Features

- Data loading from processed metadata files
- Multiple model architectures (ResNet50, MobileNetV2, custom CNN) 
- Training with automatic early stopping and learning rate reduction
- Visualization tools for training history, predictions, and data distribution
- Real-time inference using webcam

## Environment Setup
To set up this project:
1. Create a virtual environment (optional but recommended).
2. Install required dependencies listed under "Required Packages" using `pip install -r requirements.txt` or manually with pip.

## Usage

### Training a Model

```bash
# Train on the first available dataset
python train.py

# List all available datasets
python train.py --list_datasets

# Train on a specific dataset 
python train.py --dataset_id 00098

# Train a custom model (lighter, no pretrained weights)
python train.py --model_type custom --epochs 100

# Train MobileNet model (faster than ResNet50)
python train.py --model_type mobilenet

# Disable GPU or mixed precision if needed
python train.py --disable_gpu --disable_mixed_precision
```

```

## Required Packages

- tensorflow >= 2.0
- opencv-python
- numpy
- matplotlib
- scikit-learn

## Dataset Structure
Below is an example of the directory layout:
```
```
dataset/
 └── participant_001/
     ├── frames/
     ├── metadata.json
     ├── ...
 └── participant_002/
     ├── frames/
     ├── metadata.json
     ├── ...
```
```
Each `metadata.json` file references frames and associated annotations.

## License
This project is made available under the MIT License. See the LICENSE file for details.

