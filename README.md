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
9. [Evaluation](#evaluation)
10. [Real-time Prediction](#real-time-prediction)
11. [Troubleshooting](#troubleshooting)
12. [Contact](#contact)
13. [Detailed Command-Line Options](#detailed-command-line-options)

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

---

## Directory Structure
```
eye_gaze/
├── src/
│   ├── utils/                    # Utility functions
│   │   ├── mediapipe_utils.py    # Face and eye detection using MediaPipe
│   │   ├── camera_utils.py       # Camera handling utilities
│   ├── pre_processing/           # Data preprocessing pipeline
│   │   ├── extraction.py         # For extracting images and annotations
│   │   ├── process_metadata.py   # Processing metadata
│   │   ├── data_splitter.py      # Splitting data into train/test sets
│   ├── model/                    # Model definitions
│   │   ├── resnet50.py           # ResNet50-based eye gaze model
│   │   ├── gazenet.py            # GazeNet model using EfficientNetB3
│   │   ├── hybrid_model.py       # Hybrid model combining approaches
│   ├── data/                     # Dataset handling
│   │   ├── dataset.py            # Dataset class for loading and preprocessing
│   ├── test/                     # Testing and evaluation scripts
│   │   ├── predict.py            # For making predictions with trained models
│   │   ├── evaluate.py           # For evaluating model performance
│   ├── train_resnet.py           # Training script for ResNet model
│   ├── train_gazenet.py          # Training script for GazeNet model
│   ├── train_hybrid.py           # Training script for Hybrid model
│   ├── model_compare.py          # For comparing different model architectures
│   ├── preprocess_main.py        # Main script for preprocessing pipeline
├── requirements.txt              # Project dependencies
├── README.md                     # This file
```
---

## Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- MediaPipe
- Other dependencies listed in `requirements.txt`

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Dhruv-413/Eye-Gaze-Tracking-.gitt
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
1. Process the metadata:
   ```bash
   python src/pre_processing/process_metadata.py
   ```
2. Split the data into training and testing sets:
   ```bash
   python src/pre_processing/data_splitter.py
   ```

---

## Model Training
1. Train the ResNet model:
   ```bash
   python src/train_resnet.py
   ```
2. Train the GazeNet model:
   ```bash
   python src/train_gazenet.py
   ```
3. Train the Hybrid model:
   ```bash
   python src/train_hybrid.py
   ```

---

## Evaluation
1. Evaluate the trained models:
   ```bash
   python src/test/evaluate.py
   ```
2. Compare different model architectures:
   ```bash
   python src/model_compare.py
   ```

---

## Real-time Prediction
The project provides tools to make predictions with trained models on new images or in real-time applications.

### Single Image Prediction
To make predictions on individual images:

```bash
python src/test/predict.py --model_path path/to/trained_model.h5 \
                          --left_eye path/to/left_eye.jpg \
                          --right_eye path/to/right_eye.jpg \
                          --face path/to/face.jpg \
                          --output prediction_result.jpg
```

#### Parameters:
- `--model_path`: Path to the trained model (.h5 file)
- `--left_eye`: Path to the left eye image
- `--right_eye`: Path to the right eye image
- `--face`: Path to the face image
- `--metadata`: (Optional) Path to a JSON file with metadata
- `--output`: Path to save the visualization (default: gaze_prediction.jpg)
- `--img_size`: Size of input images (default: 224)

### Integration with Your Applications
The modules in `utils/mediapipe_utils.py` and `utils/camera_utils.py` provide functionality that you can integrate into your own applications:

1. **Face and Eye Detection**: The `FaceMeshDetector` class in `mediapipe_utils.py` can detect facial landmarks and extract eye regions.
2. **Camera Handling**: The `CameraCapture` class in `camera_utils.py` provides an easy interface for webcam access.

Example integration:
```python
from utils.mediapipe_utils import FaceMeshDetector
from utils.camera_utils import CameraCapture
import cv2

# Initialize the camera and face detector
with CameraCapture(camera_id=0) as camera:
    with FaceMeshDetector() as detector:
        while True:
            # Get frame from camera
            ret, frame = camera.read()
            if not ret:
                break
                
            # Process frame to find faces and eyes
            results = detector.process_frame(frame)
            left_eye_data, right_eye_data = detector.extract_eye_data(results, frame.shape)
            
            # Draw landmarks for visualization
            frame_with_landmarks = detector.draw_landmarks(frame, results)
            
            # Display the result
            cv2.imshow("Eye Tracking", frame_with_landmarks)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
```

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

---

## Model Architectures

The project implements three main model architectures:

1. **ResNet50-based Model** (`src/model/resnet50.py`)
   - Base model: ResNet50 pre-trained on ImageNet
   - Multiple inputs: face image, left eye, right eye, and metadata
   - Feature fusion: concatenation of features from all inputs
   - Output: 2D gaze coordinates (x, y)
   - Strengths: Good accuracy, robust to various face orientations

2. **GazeNet Model** (`src/model/gazenet.py`)
   - Base model: EfficientNetB3 pre-trained on ImageNet
   - Multi-task learning: gaze estimation and pose estimation
   - Enhanced feature extraction for eye regions
   - Strengths: More efficient, better for mobile applications

3. **Hybrid Model** (`src/model/hybrid_model.py`)
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
- `--split_dir <path>`: Directory to save train/test split data (default: "output/split_data")
- `--test_size <float>`: Proportion of data to use for testing (default: 0.3)
- `--skip_extraction`: Skip image extraction step
- `--skip_metadata_processing`: Skip metadata processing step
- `--skip_splitting`: Skip train/test splitting step

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

#### Data Splitting (`data_splitter.py`)
```bash
python src/pre_processing/data_splitter.py [options]
```

**Options:**
- `--input_directory <path>`: Directory with processed data (default: "output")
- `--output_directory <path>`: Directory to save split data (default: "split_data")
- `--test_split <float>`: Test split ratio (default: 0.3)

### Model Training

#### ResNet50 Training (`train_resnet.py`)
```bash
python src/train_resnet.py [options]
```

**Options:**
- `--data_dir <path>`: Root directory containing the dataset (default: ".")
- `--output_dir <path>`: Directory to save models and results (default: "./trained_models")
- `--batch_size <int>`: Batch size for training (default: 32)
- `--epochs <int>`: Number of epochs to train for (default: 50)
- `--img_size <int>`: Size of input images (square) (default: 224)
- `--val_split <float>`: Fraction of data to use for validation (default: 0.2)

#### GazeNet Training (`train_gazenet.py`)
```bash
python src/train_gazenet.py [options]
```

**Options:**
- `--train <path>`: Path to training data (required)
- `--val <path>`: Path to validation data (required)
- `--output <path>`: Output directory (default: "./output")
- `--batch-size <int>`: Batch size (default: 32)
- `--epochs <int>`: Number of epochs (default: 50)
- `--lr <float>`: Learning rate (default: 0.0005)
- `--image-size <int>`: Input image size (default: 224)
- `--mixed-precision`: Use mixed precision training
- `--no-pose`: Disable pose estimation

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

### Evaluation and Prediction

#### Model Evaluation (`evaluate.py`)
```bash
python src/test/evaluate.py [options]
```

**Options:**
- `--model_path <path>`: Path to the trained model (.h5 file) (required)
- `--data_dir <path>`: Root directory containing the dataset (default: ".")
- `--output_dir <path>`: Directory to save evaluation results (default: "./evaluation_results")
- `--batch_size <int>`: Batch size for evaluation (default: 32)
- `--img_size <int>`: Size of input images (square) (default: 224)
- `--val_split <float>`: Fraction of data to use for validation (default: 0.2)

#### Model Prediction (`predict.py`)
```bash
python src/test/predict.py [options]
```

**Options:**
- `--model_path <path>`: Path to the trained model (.h5 file) (required)
- `--left_eye <path>`: Path to the left eye image (required)
- `--right_eye <path>`: Path to the right eye image (required)
- `--face <path>`: Path to the face image (required)
- `--metadata <path>`: Path to JSON file with metadata (optional)
- `--output <path>`: Path to save the visualization (default: "gaze_prediction.jpg")
- `--img_size <int>`: Size of input images (default: 224)

#### Model Comparison (`model_compare.py`)
```bash
python src/model_compare.py [options]
```

**Options:**
- `--resnet_model <path>`: Path to trained ResNet50 model (.h5 file)
- `--efficientnet_model <path>`: Path to trained EfficientNet model (.h5 file)
- `--hybrid_model <path>`: Path to trained Hybrid model (.h5 file)
- `--data_dir <path>`: Root directory containing the dataset (default: ".")
- `--output_dir <path>`: Directory to save comparison results (default: "./model_comparison")
- `--batch_size <int>`: Batch size for evaluation (default: 32)
- `--img_size <int>`: Size of input images (square) (default: 224)
- `--val_split <float>`: Fraction of data to use for validation (default: 0.2)

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
