# Eye Gaze Estimation with ResNet50

This project implements an eye gaze estimation model using ResNet50 as a backbone. The model takes left eye, right eye, and face images as input, along with metadata features, to predict gaze coordinates on a screen.

## Project Structure

- `model.py`: Defines the ResNet50-based model architecture for eye gaze estimation
- `dataset.py`: Handles loading and preprocessing the eye gaze dataset
- `train.py`: Script for training the model
- `evaluate.py`: Script for evaluating the trained model
- `predict.py`: Script for making predictions with a trained model

## Requirements

- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

You can install the required packages with:

```
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Processing Pipeline

The pipeline consists of three main steps:

1. **Data Extraction**: Extract images and annotations from the raw dataset
2. **Metadata Processing**: Process metadata to filter invalid frames, normalize coordinates, etc.
3. **Data Splitting**: Split the processed data into training and test sets

### Running the Full Pipeline

To process raw data, generate metadata, and create train/test splits:

```
python src/preprocess_main.py --raw_data_dir dataset --processed_dir output --split_dir split_data
```

### Running Individual Steps

To only run the extraction step:
```
python src/preprocess_main.py --raw_data_dir dataset --processed_dir output --skip_metadata_processing --skip_splitting
```

To only run the metadata processing step:
```
python src/preprocess_main.py --processed_dir output --skip_extraction --skip_splitting
```

To only run the train/test splitting step:
```
python src/preprocess_main.py --processed_dir output --split_dir output/split_data --skip_extraction --skip_metadata_processing
```

### Command-line Arguments

- `--raw_data_dir`: Directory containing raw data (default: "dataset")
- `--processed_dir`: Directory to save processed data (default: "output")
- `--split_dir`: Directory to save train/test split data (default: "output/split_data")
- `--test_size`: Proportion of data to use for testing (default: 0.3)
- `--skip_extraction`: Skip the extraction step
- `--skip_metadata_processing`: Skip the metadata processing step
- `--skip_splitting`: Skip the train/test splitting step

## Directory Structure

- `dataset/`: Raw dataset (organized by subject folders)
- `output/`: Processed data (extracted images and metadata)
- `split_data/`: Train and test split data
- `src/`: Source code
  - `pre_processing/`: Data preprocessing modules
  - `utils/`: Utility functions

## Dataset Format

The dataset should follow the structure observed in the processed metadata files:

- Root directory containing `output` folder
- Under `output`, subdirectories like `00001`, `00002`, etc.
- Each subdirectory contains:
  - `processed_metadata.json`: JSON file with metadata and file references
  - `faces/`: Directory with face images
  - `left_eyes/`: Directory with left eye images
  - `right_eyes/`: Directory with right eye images

## Training

### Basic ResNet50 Model

To train the ResNet50-based model:

```
python train.py --data_dir /path/to/dataset --output_dir ./trained_models --batch_size 32 --epochs 50
```

Additional arguments:
- `--img_size`: Size of input images (default: 224)
- `--val_split`: Validation split ratio (default: 0.2)

### GazeNet Model (EfficientNetB3)

To train the GazeNet model based on EfficientNetB3:

```
python train_gazenet.py --train /path/to/train_data --val /path/to/val_data --output ./output --batch-size 32 --epochs 50
```

### Hybrid ResNet-GazeNet Model

To train the hybrid model that combines ResNet50 for facial features and EfficientNetB3 for eye-specific features:

```
python train_hybrid.py --data_dir /path/to/dataset --output_dir ./trained_models/hybrid --batch_size 32 --epochs 50 --fusion feature
```

Additional arguments:
- `--mixed_precision`: Enable mixed precision training for faster training on compatible GPUs
- `--no_pose`: Disable pose estimation auxiliary task
- `--fusion`: Choose fusion method ('feature' or 'decision')

## Evaluation

To evaluate a trained model:

```
python evaluate.py --model_path ./trained_models/model_best.h5 --data_dir /path/to/dataset --output_dir ./evaluation_results
```

This will generate:
- Performance metrics (MAE, MSE, angular error)
- Visualizations of predictions vs. ground truth
- Error heatmap across the screen
- Histogram of angular errors

## Model Comparison

To compare different model architectures:

```
python model_compare.py --resnet_model ./models/resnet_model.h5 --efficientnet_model ./models/gazenet_model.h5 --hybrid_model ./models/hybrid_model.h5 --data_dir /path/to/dataset --output_dir ./comparison_results
```

This will generate:
- Performance metrics for each model
- Comparative visualizations of model performance
- Inference time and model size comparisons

## Prediction

To make a prediction on new images:

```
python predict.py --model_path ./trained_models/model_best.h5 --left_eye path/to/left_eye.jpg --right_eye path/to/right_eye.jpg --face path/to/face.jpg --output gaze_prediction.jpg
```

Optional arguments:
- `--metadata`: Path to JSON file with metadata
- `--img_size`: Size of input images (default: 224)

## Model Architectures

### ResNet50 Model
The basic model uses ResNet50 (pre-trained on ImageNet) to extract features from the left eye, right eye, and face images separately. These features are then concatenated with metadata features and passed through fully connected layers to predict the gaze coordinates.

### GazeNet Model
GazeNet uses EfficientNetB3 as the backbone and incorporates an auxiliary task for head pose estimation to improve the main task of gaze prediction.

### Hybrid Model
The hybrid model combines the strengths of both architectures:
- ResNet50 for capturing detailed facial textures and structures
- EfficientNetB3 (GazeNet approach) for eye-specific features
- Two fusion methods:
  - Feature-level fusion: Combines features before prediction layers
  - Decision-level fusion: Makes separate predictions and combines them

## Performance Metrics

The models are evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Angular Error (in degrees)
- Inference time per sample
- Model size

## License

This project is provided for educational and research purposes only.
