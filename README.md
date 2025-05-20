# Eye Gaze Tracking System

## Introduction
This repository provides a comprehensive pipeline for eye gaze estimation and tracking. The system enables precise eye gaze detection through advanced deep learning techniques, supporting applications in human-computer interaction, accessibility solutions, and attention analysis.

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Directory Structure](#directory-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Dataset Preparation](#dataset-preparation)
7. [Data Processing](#data-processing)
8. [Model Training](#model-training)
9. [Real-time Cursor Control](#real-time-cursor-control)
10. [Troubleshooting](#troubleshooting)
11. [Model Architectures](#model-architectures)
12. [Future Work](#future-work)
13. [Contact](#contact)
14. [Detailed Command-Line Options](#detailed-command-line-options)

---

## Overview
This project provides an end-to-end solution for gaze-based interaction. By leveraging advanced computer vision methods and deep learning architectures, it enables accurate gaze estimation and real-time tracking. The pipeline includes:
- Data preprocessing and normalization
- Multiple model architectures for gaze estimation
- Training and evaluation frameworks
- Real-time detection using MediaPipe face mesh
- Comprehensive visualization tools for analysis

---

## Key Features
- **Multiple model architectures**: ResNet50, GazeNet (EfficientNetB3), and a hybrid model
- **Multi-input modeling**: Face, left/right eyes, and metadata integration
- **Comprehensive preprocessing pipeline**: Image extraction, metadata processing, and dataset splitting
- **Real-time capabilities**: Face and eye detection using MediaPipe
- **Evaluation tools**: Angular error calculation, visualization of predictions
- **Optimization techniques**: Mixed precision training, early stopping, and learning rate scheduling
- **Real-time cursor control**: Automatic cursor movement based on eye gaze direction

---

## Directory Structure
```
eye_gaze/
├── src/
│   ├── utils/                    # Utility functions
│   │   ├── mediapipe_utils.py    # Face and eye detection using MediaPipe
│   │   ├── camera_utils.py       # Camera handling utilities
│   │   ├── metadata_utils.py     # Metadata generation and processing
│   │   ├── gaze_processing.py    # Gaze direction processing utilities
│   │   ├── calibration_utils.py  # Calibration utilities for gaze tracking
│   ├── pre_processing/           # Data preprocessing pipeline
│   │   ├── extraction.py         # For extracting images and annotations
│   │   ├── process_metadata.py   # Processing metadata
│   ├── model/                    # Model definitions
│   │   ├── hybrid_model.py       # Hybrid model combining approaches
│   ├── data/                     # Dataset handling
│   │   ├── dataset.py            # Dataset class for loading and preprocessing
│   ├── train_hybrid.py           # Training script for Hybrid model
│   ├── preprocess_main.py        # Main script for preprocessing pipeline
│   ├── main.py                   # Main application script for gaze tracking
│   ├── main2.py                  # Simplified real-time gaze to cursor control
├── requirements.txt              # Project dependencies
├── README.md                     # This file
```
---

## Prerequisites
- Python 3.9 or 3.10
- TensorFlow 2.x
- OpenCV
- MediaPipe
- Other dependencies listed in `requirements.txt`

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Dhruv-413/Eye-Gaze-Tracking-
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate       # For Linux/macOS
   venv\Scripts\activate          # For Windows
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation
1. Download the dataset and place it in the `data/` directory.
2. Run the preprocessing script to extract images and annotations:
   ```bash
   python src/pre_processing/extraction.py
   ```

---

## Data Processing
1. Process the images and metadata:
   ```bash
   python src/preprocess_main.py
   ```
---

## Model Training
1. Train the Hybrid model:
   ```bash
   python src/train_hybrid.py
   ```

---

## Real-time Cursor Control

The project includes a simplified script (`main2.py`) for real-time gaze-based cursor control:

```bash
python src/main2.py --model_path path/to/your/model.h5 --cam_index 0
```

#### Parameters:
- `--model_path`: Path to the trained model (.h5 file) (required)
- `--cam_index`: Webcam index to use (default: 0)

This script:
1. Captures webcam input
2. Detects face and eye landmarks using MediaPipe
3. Processes the detected features with your trained model
4. Moves the mouse cursor based on the predicted gaze direction

---

## Troubleshooting

### Installation Issues
1. **TensorFlow GPU Issues**:
   - Ensure compatible versions of CUDA and cuDNN are installed
   - Check compatibility matrix: [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
   - Verify GPU detection: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

2. **MediaPipe Installation Issues**:
   - If you encounter "ImportError: DLL load failed", reinstall MediaPipe: `pip uninstall mediapipe && pip install mediapipe`
   - On Linux, ensure required libraries are installed: `sudo apt-get install libgl1-mesa-glx`

3. **Package Conflicts**:
   - Use a clean virtual environment: `python -m venv fresh_venv`
   - Install packages in order: TensorFlow first, then other dependencies

### Dataset Issues
1. **Missing Files Error**:
   - Ensure all required JSON files exist in each subject folder
   - Check file permissions: read/write access to dataset directories
   - Verify file formats match the expected structures

2. **Preprocessing Failures**:
   - Run with verbose logging: add `--verbose` to preprocessing commands
   - Process one subject at a time to isolate issues
   - Check disk space for output directories

### Training Issues
1. **Out of Memory Errors**:
   - Reduce batch size: `--batch_size 8` or even smaller
   - Enable memory growth: set environment variable `TF_FORCE_GPU_ALLOW_GROWTH=true`
   - Close other GPU applications while training

2. **Poor Model Performance**:
   - Check training/validation loss curves for overfitting
   - Try different learning rates: `--lr 0.0001` or `--lr 0.00001`
   - Increase regularization or dropout if overfitting
   - Ensure proper data normalization and preprocessing

3. **Slow Training**:
   - Enable mixed precision: `--mixed_precision`
   - Use a smaller model architecture for initial experiments
   - Reduce input image dimensions: `--img_size 160`

### Prediction Issues
1. **Incorrect Gaze Predictions**:
   - Verify input image formats (RGB vs BGR)
   - Check image preprocessing steps match training
   - Ensure face is fully visible and well-lit

2. **MediaPipe Detection Failures**:
   - Improve lighting conditions
   - Adjust minimum detection confidence: `min_detection_confidence=0.6`
   - For glasses, try `refine_landmarks=True` in FaceMeshDetector
   
3. **Cursor Control Issues**:
   - Run calibration process before prediction
   - Adjust sensitivity parameters if cursor moves too fast/slow
   - Ensure stable lighting conditions for consistent detection

---

## Model Architectures

1. **Hybrid Model** (`src/model/hybrid_model.py`)
   - Combines ResNet50 and EfficientNetB3 architectures
   - Two fusion methods: feature-level and decision-level
   - Integrates face, eye, and metadata features
   - Strengths: Higher accuracy, especially for challenging cases

---

## Future Work
- Integration with reinforcement learning for gaze target prediction
- Mobile device optimization
- Web-based demonstration and API
- Support for low-resource environments
- Additional data augmentation techniques

---

## Contact

### Project Information
- **Project Link:** [https://github.com/Dhruv-413/Eye-Gaze-Tracking-](https://github.com/Dhruv-413/Eye-Gaze-Tracking-)

### Author
- **Dhruv**: [GitHub Profile](https://github.com/Dhruv-413)
- **Email:** Contact through GitHub for email information

---

## Detailed Command-Line Options

### Preprocessing Pipeline

#### Complete Pipeline (`preprocess_main.py`)
```bash
python src/preprocess_main.py [options]
```

**Options:**
- `--raw_data_dir <path>`: Directory containing raw data (default: "dataset")
- `--processed_dir <path>`: Directory to save processed data (default: "output")
- `--test_size <float>`: Proportion of data to use for testing (default: 0.3)
- `--skip_extraction`: Skip image extraction step
- `--skip_metadata_processing`: Skip metadata processing step

#### Image Extraction (`extraction.py`)
```bash
python src/pre_processing/extraction.py [options]
```

**Options:**
- `--parent_dir <path>`: Parent directory containing subject folders (default: "dataset")
- `--output_dir <path>`: Output directory for extracted data (default: "output")

#### Metadata Processing (`process_metadata.py`)
```bash
python src/pre_processing/process_metadata.py [options]
```

**Options:**
- `--output_dir <path>`: Directory containing processed data (default: "output")


### Model Training

#### Hybrid Model Training (`train_hybrid.py`)
```bash
python src/train_hybrid.py [options]
```

**Options:**
- `--data_dir <path>`: Directory containing the dataset (required)
- `--output_dir <path>`: Output directory (default: "./output/hybrid_model")
- `--batch_size <int>`: Batch size (default: 32)
- `--epochs <int>`: Number of epochs (default: 50)
- `--lr <float>`: Learning rate (default: 0.0005)
- `--image_size <int>`: Input image size (default: 224)
- `--mixed_precision`: Use mixed precision training
- `--no_pose`: Disable pose estimation
- `--fusion <method>`: Feature fusion method (choices: "feature", "decision") (default: "feature")
- `--val_split <float>`: Validation split ratio (default: 0.2)

### Main Application

#### Gaze Tracking Application (`main.py`)
```bash
python src/main.py [options]
```

**Options:**
- `--model_path <path>`: Path to the trained model (.h5 file) (required)
- `--camera_id <int>`: Camera device ID (default: 0)
- `--width <int>`: Camera capture width (default: 640)
- `--height <int>`: Camera capture height (default: 480)
- `--display_landmarks`: Display facial landmarks on frame
- `--fullscreen`: Run in fullscreen mode
- `--record`: Record the session
- `--output_dir <path>`: Directory to save recordings (default: "./recordings")

#### Real-time Cursor Control (`main2.py`)
```bash
python src/main2.py [options]
```

**Options:**
- `--model_path <path>`: Path to the trained model (.h5 file) (required)
- `--cam_index <int>`: Webcam index (default: 0)

### MediaPipe Face Mesh Detector Options

The `FaceMeshDetector` class in `mediapipe_utils.py` accepts the following parameters:

- `static_image_mode=False`: Set to True for static images, False for video streams
- `max_num_faces=1`: Maximum number of faces to detect
- `refine_landmarks=True`: Whether to refine landmarks around eyes and lips
- `min_detection_confidence=0.7`: Minimum confidence value for face detection
- `min_tracking_confidence=0.5`: Minimum confidence value for face tracking

Example usage with custom parameters:
```python
from utils.mediapipe_utils import FaceMeshDetector

detector = FaceMeshDetector(
    max_num_faces=2,
    min_detection_confidence=0.6,
    refine_landmarks=True
)
```

### Camera Capture Options

The `CameraCapture` class in `camera_utils.py` accepts the following parameters:

- `camera_id=0`: Camera device ID (default: 0 for primary webcam)
- `width=640`: Capture width in pixels
- `height=480`: Capture height in pixels
- `fps=30`: Target frames per second

Example usage with custom parameters:
```python
from utils.camera_utils import CameraCapture

camera = CameraCapture(
    camera_id=1,  # Use second camera
    width=1280,
    height=720,
    fps=60
)
```
