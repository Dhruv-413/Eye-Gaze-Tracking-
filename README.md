
## Overview
This project focuses on gaze detection and gesture detection using EEG and BCI data. It includes modules for data extraction, preprocessing, model training, and visualization.
- **Gaze Tracking**: Train and evaluate models for gaze point prediction.
## Featuresocessing**: Extract and preprocess images with annotations.
- **Gaze Tracking**: Train and evaluate models for gaze point prediction.
- **Data Processing**: Extract and preprocess images with annotations.
- **Visualization**: Analyze dataset distributions and model predictions.
- GPU with CUDA support (recommended for training)
## Architectureprocessing and inference
This repository is organized into multiple modules for data processing, model training, and visualization:
- **Data Processing**: Handles image extraction and annotation via scripts in `src/Data_processing`.
- **Model**: Contains training scripts for gaze tracking, model evaluation, and multi-dataset support in `src/model`.- Python 3.7 or higher
- **Visualization**: Provides scripts to visualize dataset distributions and model predictions.depending on the model implementation)
isted in `requirements.txt`
## Hardware Requirements
- GPU with CUDA support (recommended for training)
- Basic CPU for data preprocessing and inference1. Clone the repository:

## Software Requirements-url>
- Python 3.7 or higherEEG-Gaze-Detection-Gesture-Detection
- TensorFlow / PyTorch (depending on the model implementation)
- Other dependencies listed in `requirements.txt`
tall dependencies:
## Setup Instructions   ```bash
1. Clone the repository:ements.txt
   ```bash
   git clone <repository-url>
   cd BCI-EEG-Gaze-Detection-Gesture-Detectionure you have the required datasets in the appropriate directory structure.
   ```

2. Install dependencies:### Data Processing
   ```bashscript to extract images and annotations:
   pip install -r requirements.txt
   ```notations.py

3. Ensure you have the required datasets in the appropriate directory structure.
 Model Training
## UsageTrain the gaze tracking model:
### Data Processing
Run the script to extract images and annotations:.py --metadata <path-to-metadata> --epochs 50
```bash
python src/Data_processing/extract_images_with_annotations.py
```in on multiple datasets:
```bash
### Model Trainingi_dataset.py --metadata_paths <path1,path2,...>
Train the gaze tracking model:
```bash
python src/model/gaze_tracking.py --metadata <path-to-metadata> --epochs 50 Visualization
```Visualize dataset distributions:

Train on multiple datasets:ets.py --dataset_dir <path-to-datasets>
```bash
python src/model/train_multi_dataset.py --metadata_paths <path1,path2,...>
```Directory Structure
- `src/Data_processing`: Scripts for data extraction and preprocessing.
### Visualization for model training, evaluation, and visualization.
Visualize dataset distributions:
```bash
python src/model/visualize_datasets.py --dataset_dir <path-to-datasets>- Explore more sophisticated neural network architectures
```vanced EEG signal processing techniques

## Repository Layout
- `src/Data_processing`: Main scripts for data extraction, augmentation, and annotation.
- `src/model`: Model training, configuration, and evaluation. It includes GPU/CPU management and multi-dataset handling.Contributions are welcome! Please fork the repository and submit a pull request.
- `requirements.txt`: Lists all dependencies needed to run the project.
- `utils`: Additional utility functions for logging, image processing, etc.
This project is licensed under the MIT License.## Future Work- Explore more sophisticated neural network architectures
- Integrate advanced EEG signal processing techniques
- Implement real-time gaze tracking and gesture detection

## Additional Resources
For any questions or troubleshooting tips, refer to:
- **Issues & Discussions**: Use GitHub to submit issues, share suggestions, or ask for help.
- **Official Documentation**: Review the TensorFlow or PyTorch docs if you encounter compatibility issues.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
