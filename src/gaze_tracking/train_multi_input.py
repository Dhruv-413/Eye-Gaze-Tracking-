#!/usr/bin/env python3
"""
Training script for multi-input gaze tracking model using face and eye images
with MediaPipe for detection
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data_loader import GazeDataLoader
from models.head_pose_gaze_model import create_head_pose_gaze_model
from visualization import plot_training_history, visualize_predictions
from gpu_utils import configure_gpu, force_gpu_usage

def train_multi_input_model(dataset_paths, args):
    """
    Train a multi-input model with face and eye images
    
    Args:
        dataset_paths: List of paths to metadata files
        args: Command line arguments
    """
    all_face_images = []
    all_left_eye_images = []
    all_right_eye_images = []
    all_gaze_points = []
    
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
                
                all_face_images.append(image_dict['faces'])
                all_left_eye_images.append(image_dict['left_eyes'])
                all_right_eye_images.append(image_dict['right_eyes'])
                all_gaze_points.append(gaze_points)
            else:
                print(f"Dataset doesn't have eye images - skipping")
            
        except Exception as e:
            print(f"Error loading dataset {metadata_path}: {e}")
    
    if not all_face_images:
        print("No valid data loaded from any dataset.")
        return
    
    # Combine all datasets
    face_images = np.concatenate(all_face_images, axis=0)
    left_eye_images = np.concatenate(all_left_eye_images, axis=0)
    right_eye_images = np.concatenate(all_right_eye_images, axis=0)
    gaze_points = np.concatenate(all_gaze_points, axis=0)
    
    print(f"Combined dataset: {face_images.shape[0]} samples")
    print(f"Face image shape: {face_images.shape}")
    print(f"Left eye image shape: {left_eye_images.shape}")
    print(f"Right eye image shape: {right_eye_images.shape}")
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
        'right_eye_input': right_eye_images[train_idx]
    }
    
    X_val = {
        'face_input': face_images[val_idx],
        'left_eye_input': left_eye_images[val_idx],
        'right_eye_input': right_eye_images[val_idx]
    }
    
    X_test = {
        'face_input': face_images[test_idx],
        'left_eye_input': left_eye_images[test_idx],
        'right_eye_input': right_eye_images[test_idx]
    }
    
    y_train = gaze_points[train_idx]
    y_val = gaze_points[val_idx]
    y_test = gaze_points[test_idx]
    
    print(f"Data splits - Train: {len(train_idx)}, "
          f"Validation: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create model
    print(f"Creating multi-input model...")
    model = create_head_pose_gaze_model(
        input_shape=(64, 64, 3),
        use_mixed_precision=args.mixed_precision
    )
    
    model.summary()
    
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
            filepath=os.path.join(args.output_dir, 'checkpoints', 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
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
    parser = argparse.ArgumentParser(description="Train multi-input gaze tracking model")
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
