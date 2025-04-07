#!/usr/bin/env python3
"""
Training script for multi-input gaze tracking model using face and eye images
with MediaPipe for detection plus head pose and metadata
"""

import os
import argparse
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data_loader import GazeDataLoader
from models.head_pose_gaze_model import create_head_pose_gaze_model, get_training_callbacks
from visualization import plot_training_history, visualize_predictions
from gpu_utils import configure_gpu, force_gpu_usage

def extract_head_pose_from_metadata(metadata_file, indices=None):
    """
    Extract head pose angles (pitch, yaw, roll) from metadata file.
    
    Args:
        metadata_file: Path to metadata JSON file
        indices: Optional indices to extract (for specific frames)
    
    Returns:
        np.ndarray of shape (N, 3) with [pitch, yaw, roll] values
    """
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    # Use specified indices or all data
    if indices is not None:
        samples = [data[i] for i in indices if i < len(data)]
    else:
        samples = data
    
    head_pose = []
    for sample in samples:
        # Default values if data is missing
        pitch, yaw, roll = 0.0, 0.0, 0.0
        
        # Try to extract from motion_data if available
        if 'motion_data' in sample and isinstance(sample['motion_data'], dict):
            motion = sample['motion_data']
            if 'AttitudePitch' in motion:
                pitch = motion['AttitudePitch']
            if 'AttitudeYaw' in motion:
                yaw = motion['AttitudeYaw']
            if 'AttitudeRoll' in motion:
                roll = motion['AttitudeRoll']
                
        head_pose.append([pitch, yaw, roll])
    
    return np.array(head_pose, dtype=np.float32)

def extract_metadata_features(metadata_file, indices=None, feature_count=6):
    """
    Extract additional metadata features for gaze prediction.
    
    Args:
        metadata_file: Path to metadata JSON file
        indices: Optional indices to extract
        feature_count: Number of features to extract
    
    Returns:
        np.ndarray of shape (N, feature_count) with metadata features
    """
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    # Use specified indices or all data
    if indices is not None:
        samples = [data[i] for i in indices if i < len(data)]
    else:
        samples = data
    
    metadata_features = []
    for sample in samples:
        features = np.zeros(feature_count, dtype=np.float32)
        
        # Extract screen info if available
        if 'screen_data' in sample and isinstance(sample['screen_data'], dict):
            screen = sample['screen_data']
            if 'W' in screen and 'H' in screen:
                # Add screen dimensions and aspect ratio
                features[0] = screen.get('W', 0) / 1000.0  # Normalize width
                features[1] = screen.get('H', 0) / 1000.0  # Normalize height
                if features[1] > 0:
                    features[2] = features[0] / features[1]  # Aspect ratio
        
        # Extract face position from face_grid if available
        if 'face_grid_data' in sample and isinstance(sample['face_grid_data'], dict):
            face_grid = sample['face_grid_data']
            if all(k in face_grid for k in ['X', 'Y', 'W', 'H']):
                # Normalized face grid position and size
                features[3] = face_grid.get('X', 0) / 30.0  # Typical grid range
                features[4] = face_grid.get('Y', 0) / 30.0
                features[5] = face_grid.get('W', 0) * face_grid.get('H', 0) / 400.0  # Area
        
        metadata_features.append(features)
    
    return np.array(metadata_features, dtype=np.float32)

def train_multi_input_model(dataset_paths, args):
    """
    Train a multi-input model with face, eye images, head pose, and metadata
    
    Args:
        dataset_paths: List of paths to metadata files
        args: Command line arguments
    """
    all_face_images = []
    all_left_eye_images = []
    all_right_eye_images = []
    all_gaze_points = []
    all_head_pose = []
    all_metadata = []
    
    print(f"Loading data from {len(dataset_paths)} datasets...")
    
    # Load and combine data from all datasets
    for metadata_path in dataset_paths:
        try:
            print(f"\nProcessing dataset: {os.path.basename(os.path.dirname(metadata_path))}")
            data_loader = GazeDataLoader(metadata_path, img_size=(64, 64))
            image_dict, gaze_points = data_loader.load_data(include_eyes=True)
            
            # Only include samples that have all three images
            if 'left_eyes' in image_dict and 'right_eyes' in image_dict:
                print(f"Loaded {len(image_dict['faces'])} complete samples from dataset")
                
                # Extract indices for current dataset samples (just 0 to n-1)
                sample_indices = np.arange(len(image_dict['faces']))
                
                # Extract head pose and metadata for these samples
                head_pose = extract_head_pose_from_metadata(metadata_path, sample_indices)
                metadata = extract_metadata_features(metadata_path, sample_indices)
                
                print(f"Extracted head pose data: {head_pose.shape}, metadata features: {metadata.shape}")
                
                all_face_images.append(image_dict['faces'])
                all_left_eye_images.append(image_dict['left_eyes'])
                all_right_eye_images.append(image_dict['right_eyes'])
                all_gaze_points.append(gaze_points)
                all_head_pose.append(head_pose)
                all_metadata.append(metadata)
            else:
                print(f"Dataset doesn't have eye images - skipping")
            
        except Exception as e:
            print(f"Error loading dataset {metadata_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_face_images:
        print("No valid data loaded from any dataset.")
        return
    
    # Combine all datasets
    face_images = np.concatenate(all_face_images, axis=0)
    left_eye_images = np.concatenate(all_left_eye_images, axis=0)
    right_eye_images = np.concatenate(all_right_eye_images, axis=0)
    gaze_points = np.concatenate(all_gaze_points, axis=0)
    head_pose = np.concatenate(all_head_pose, axis=0)
    metadata = np.concatenate(all_metadata, axis=0)
    
    print(f"Combined dataset: {face_images.shape[0]} samples")
    print(f"Face image shape: {face_images.shape}")
    print(f"Left eye image shape: {left_eye_images.shape}")
    print(f"Right eye image shape: {right_eye_images.shape}")
    print(f"Head pose shape: {head_pose.shape}")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Gaze points shape: {gaze_points.shape}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split data
    indices = np.arange(len(face_images))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    
    X_train = {
        'face_input': face_images[train_idx],
        'left_eye_input': left_eye_images[train_idx],
        'right_eye_input': right_eye_images[train_idx],
        'head_pose_input': head_pose[train_idx],
        'metadata_input': metadata[train_idx]
    }
    
    X_val = {
        'face_input': face_images[val_idx],
        'left_eye_input': left_eye_images[val_idx],
        'right_eye_input': right_eye_images[val_idx],
        'head_pose_input': head_pose[val_idx],
        'metadata_input': metadata[val_idx]
    }
    
    X_test = {
        'face_input': face_images[test_idx],
        'left_eye_input': left_eye_images[test_idx],
        'right_eye_input': right_eye_images[test_idx],
        'head_pose_input': head_pose[test_idx],
        'metadata_input': metadata[test_idx]
    }
    
    y_train = gaze_points[train_idx]
    y_val = gaze_points[val_idx]
    y_test = gaze_points[test_idx]
    
    print(f"Data splits - Train: {len(train_idx)}, "
          f"Validation: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create model
    print(f"Creating multi-input model with head pose and metadata...")
    model = create_head_pose_gaze_model(
        input_shape=(64, 64, 3),
        use_mixed_precision=args.mixed_precision,
        metadata_shape=metadata.shape[1]
    )
    
    model.summary()
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train the model
    try:
        print("\nStarting training...")
        history = model.fit(
            X_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot history
        plot_training_history(history, 
                           save_path=os.path.join(args.output_dir, 'training_history.png'))
        
        # Evaluate model
        print("\nEvaluating model...")
        loss, mae, mse = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test MSE: {mse:.4f}")
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Visualize predictions
        visualize_predictions(y_test, predictions, 
                           save_path=os.path.join(args.output_dir, 'predictions.png'))
        
        # Save model
        model_save_path = os.path.join(args.output_dir, args.model_path)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Train multi-input gaze tracking model with head pose and metadata")
    parser.add_argument('--dataset_dir', type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output'),
                       help='Directory containing dataset folders')
    parser.add_argument('--dataset_id', type=str, default=None,
                       help='Specific dataset ID to use (e.g. "00241")')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default="gaze_tracking_multi_input_model.h5", 
                       help='Path to save the trained model')
    parser.add_argument('--output_dir', type=str, default="model_output_multi_input",
                       help='Directory to save model outputs and visualizations')
    parser.add_argument('--use_all_datasets', action='store_true',
                       help='Use all available datasets for training')
    parser.add_argument('--force_gpu', action='store_true',
                       help='Try harder to find and use CUDA GPU')
    parser.add_argument('--metadata_features', type=int, default=6, 
                       help='Number of metadata features to use')
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Try to configure GPU
    gpu_available = configure_gpu()
    
    # If that didn't work and force_gpu is set, try harder
    if not gpu_available and args.force_gpu:
        print("Initial GPU configuration failed, trying more aggressive methods...")
        gpu_available = force_gpu_usage()
        
    # Enable mixed precision if GPU is available
    args.mixed_precision = False
    if gpu_available:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"Mixed precision enabled: {policy.name}")
            args.mixed_precision = True
        except Exception as e:
            print(f"Error enabling mixed precision: {e}")
    
    # Find available datasets
    base_dir = args.dataset_dir
    available_datasets = GazeDataLoader.find_available_datasets(base_dir)
    
    if not available_datasets:
        print(f"No datasets found in {base_dir}")
        return
        
    # Determine which datasets to use
    if args.dataset_id:
        # Specific dataset by ID
        selected_datasets = []
        for dataset in available_datasets:
            if args.dataset_id in dataset:
                selected_datasets.append(dataset)
                break
        
        if not selected_datasets:
            print(f"Dataset with ID {args.dataset_id} not found")
            return
            
        print(f"Using dataset with ID {args.dataset_id}")
    elif args.use_all_datasets:
        # Use all available datasets
        selected_datasets = available_datasets
        print(f"Using all {len(selected_datasets)} available datasets")
    else:
        # Default to first dataset
        selected_datasets = [available_datasets[0]]
        print(f"Using first available dataset: {os.path.basename(os.path.dirname(selected_datasets[0]))}")
        print("To use all datasets, add --use_all_datasets flag")
        
    # Train the multi-input model
    train_multi_input_model(selected_datasets, args)

if __name__ == "__main__":
    main()
