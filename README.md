
- **Gaze Tracking**: Train and evaluate models for gaze point prediction.

## Features
- **Data Processing**: Extract and preprocess images with annotations.
- **Gaze Tracking**: Train and evaluate models for gaze point prediction.
- **Visualization**: Analyze dataset distributions and model predictions.

## Architecture
This repository is organized into multiple modules for data processing, model training, and visualization:
- **Data Processing**: Handles image extraction and annotation via scripts in `src/Data_processing`.
- **Model**: Contains training scripts for gaze tracking, model evaluation, and multi-dataset support in `src/model`.
- **Visualization**: Provides scripts to visualize dataset distributions and model predictions.

## Hardware Requirements
- GPU with CUDA support (recommended for training)
- Basic CPU for data preprocessing and inference

## Software Requirements
- Python 3.7 or higher
- TensorFlow / PyTorch (depending on the model implementation)
- Other dependencies listed in `requirements.txt`

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd BCI-EEG-Gaze-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required datasets in the appropriate directory structure.

## Usage
### Data Processing
Run the script to extract images and annotations:
```bash
python src/Data_processing/extract_images_with_annotations.py
```

### Model Training
Train the gaze tracking model:
```bash
python src/model/gaze_tracking.py --metadata <path-to-metadata> --epochs 50
```

Train on multiple datasets:
```bash
python src/model/train_multi_dataset.py --metadata_paths <path1,path2,...>
```

### Visualization
Visualize dataset distributions:
```bash
python src/model/visualize_datasets.py --dataset_dir <path-to-datasets>
```

## Directory Structure
- `src/Data_processing`: Scripts for data extraction and preprocessing.
- `src/model`: Scripts for model training, evaluation, and visualization.
- `requirements.txt`: Lists all dependencies needed to run the project.
- `utils`: Additional utility functions for logging, image processing, etc.

## Future Work
- Explore more sophisticated neural network architectures
- Integrate advanced EEG signal processing techniques
- Implement real-time gaze tracking

## Additional Resources
For any questions or troubleshooting tips, refer to:
- **Issues & Discussions**: Use GitHub to submit issues, share suggestions, or ask for help.
- **Official Documentation**: Review the TensorFlow or PyTorch docs if you encounter compatibility issues.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
