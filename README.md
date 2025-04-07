# Gaze Tracking System (Using GazeCapture Dataset)

## Introduction
This repository provides a comprehensive pipeline for gaze estimation and tracking, built on the GazeCapture dataset. The system enables precise eye gaze detection through advanced machine learning techniques, supporting applications in human-computer interaction, accessibility solutions, and attention analysis.

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Directory Structure](#directory-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Dataset Preparation (GazeCapture)](#dataset-preparation-gazecapture)
7. [Data Processing](#data-processing)
8. [Dataset Inspection](#dataset-inspection)
9. [Model Training](#model-training)
10. [Evaluation](#evaluation)
11. [Visualization](#visualization)
12. [Usage Examples](#usage-examples)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)
15. [Contact](#contact)

---

## Overview
This project provides a comprehensive, end-to-end solution for gaze-based interaction. By leveraging advanced computer vision methods and deep learning architectures, it enables accurate gaze estimation and real-time tracking across diverse environments. It includes scripts for:
- Loading and filtering data & metadata
- Normalizing gaze points
- Training neural networks with face and eye inputs
- Handling head pose and additional metadata features
- Visualizing training history, gaze heatmaps, and coverage metrics

---

## Key Features
- Robust multi-input modeling (face, eyes, and head pose) for high-accuracy gaze detection
- Scalable data pre-processing pipeline for large datasets
- Modular code structure to facilitate easy customization
- Integrated visualization tools for analysis and debugging
- Automated data validation and cleaning procedures
- Multi-device calibration support
- Cross-platform compatibility (Windows, macOS, Linux)
- Real-time inference capabilities

---

## Directory Structure
Below is a detailed breakdown of each component in the repository:
```
repository/
├── data/                    # Dataset storage
│   ├── raw/                 # Original GazeCapture data
├── src/
│   ├── core/                # Core processing functions and shared utilities
│   │   ├── config.py        # Configuration parameters
│   │   └── ....
│   ├── Data_processing/     # Data preparation pipeline
│   │   ├── extract_images_with_annotations.py
│   │   └── process_metadata.py
│   ├── gaze_tracking/       # Primary model implementations
│   │   ├── models/          # Neural network architectures
│   │   ├── ....
│   ├── utils/               # Helper functions and utilities
│   └── interaction/         # Real-time interaction capabilities
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Prerequisites
Before installation, ensure you have the following:
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for model training)
- At least 8GB RAM (16GB+ recommended for full dataset)
- 100GB+ free disk space for the full GazeCapture dataset

---

## Installation
Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/Dhruv-413/Eye-Gaze-Tracking-.git
   cd gaze-tracking-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # For Linux/macOS
   source venv/bin/activate
   
   # For Windows
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}, GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
   ```

---

## Dataset Preparation (GazeCapture)
To use the GazeCapture dataset:

1. Request access and download from [GazeCapture](http://gazecapture.csail.mit.edu/).
2. Extract the dataset into the `data/raw/` directory.
3. Run the preprocessing scripts:
   ```bash
   # Extract frames from videos and annotations
   python src/Data_processing/extract_images_with_annotations.py --input_dir data/raw/ --output_dir data/processed/

   # Process and normalize metadata
   python src/Data_processing/process_metadata.py --input_dir data/processed/ --output_file data/processed/metadata.json
   ```

> **Note**: Processing the full GazeCapture dataset (~2.5M frames) may take several hours depending on your hardware.

---

## Data Processing
The data processing pipeline handles:

1. **Frame extraction**: Converting video data to individual frames
2. **Face detection**: Identifying and cropping facial regions
3. **Metadata validation**: Ensuring data integrity and consistency
4. **Normalization**: Standardizing gaze coordinates and image dimensions

Configuration options:
```bash
python src/Data_processing/extract_images_with_annotations.py --help
python src/Data_processing/process_metadata.py --help
```

---

## Dataset Inspection
To verify dataset integrity and explore statistics:

```bash
python src/gaze_tracking/inspect_dataset.py --input_dir data/processed/ --visualize
```

Options:
- `--visualize`: Generate visualization of dataset distribution
- `--check_paths`: Verify file existence and integrity
- `--fix_metadata`: Automatically correct common metadata issues
- `--export_stats`: Save statistical analysis to CSV

---

## Model Training
Train models with customizable parameters:

```bash
# Basic training with default parameters
python src/gaze_tracking/models/head_pose_gaze_model.py --data_dir data/processed/ --output_dir data/results/

# Advanced training options
python src/gaze_tracking/models/head_pose_gaze_model.py \
  --model_type resnet50 \
  --batch_size 64 \
  --epochs 100 \
  --learning_rate 0.0001 \
  --use_augmentation \
  --use_mixed_precision \
  --validation_split 0.15
```

Available model architectures:
- `resnet18`, `resnet34`, `resnet50` (recommended for accuracy)
- `mobilenetv2`, `efficientnet_b0` (recommended for speed)
- `custom_cnn` (smaller but faster option)

---

## Evaluation
Evaluate trained models on test data:

```bash
python src/gaze_tracking/evaluation/evaluate_model.py \
  --model_path data/results/model_best.pth \
  --test_data data/processed/test \
  --save_predictions
```

The script generates:
- Error metrics (Euclidean distance, angular error)
- Confusion matrices for classification tasks
- Precision-recall curves
- Inference speed benchmarks

---

## Visualization
Generate visualizations to analyze results:

```bash
# Generate heatmaps and error distribution plots
python src/utils/visualization.py --results_dir data/results/ --output_dir data/results/visualizations/

# Create interactive dashboard (requires Dash)
python src/utils/dashboard.py --data_dir data/results/
```

Example visualizations include:
- Gaze point heatmaps
- Error distribution histograms
- Training/validation curves
- Attention maps for model interpretability

---

## Usage Examples

### Simple Inference

```python
from src.gaze_tracking.models import GazeEstimator

# Initialize the model
model = GazeEstimator.from_pretrained("data/results/model_best.pth")

# Predict from image
import cv2
image = cv2.imread("path/to/image.jpg")
gaze_point = model.predict(image)
print(f"Predicted gaze point: {gaze_point}")
```

### Real-time Gaze Tracking

```bash
# Run the real-time demo with webcam
python src/interaction/real_time_tracker.py --model data/results/model_best.pth
```

### Batch Processing

```bash
# Process a folder of images
python src/interaction/batch_process.py --input_dir path/to/images/ --output_dir path/to/results/ --model data/results/model_best.pth
```

---

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   - Reduce batch size: `--batch_size 16`
   - Use mixed precision: `--mixed_precision=True`
   - Set memory growth: `--allow_memory_growth=True`
   - Process smaller image sizes: `--image_size 128`

2. **Dataset Loading Errors**
   - Run `inspect_dataset.py` with `--fix_metadata` flag
   - Ensure correct data directory structure
   - Check file permissions
   - Verify TFRecord format if using TensorFlow's native format

3. **Low Accuracy Issues**
   - Increase training epochs: `--epochs 200`
   - Try different model architectures: `--model_type efficientnetb0`
   - Enable data augmentation: `--use_augmentation`
   - Adjust learning rate: `--learning_rate 0.0001`
   - Try different optimizers: `--optimizer adam`

4. **Installation Issues**
   - Check TensorFlow GPU compatibility
   - Verify CUDA and cuDNN versions match TensorFlow requirements
   - Ensure all dependencies are correctly installed
   - Verify Python version (3.8+ required)
   - Run `tf.test.is_gpu_available()` to confirm GPU detection

## License
This project is available under the MIT License. Refer to the LICENSE file for complete terms.

---

## Contact

Project Link: [https://github.com/Dhruv-413/Eye-Gaze-Tracking-](https://github.com/Dhruv-413/Eye-Gaze-Tracking-)

