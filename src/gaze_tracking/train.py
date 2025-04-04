#!/usr/bin/env python3
"""
Main script to train the gaze tracking model.
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from data_loader import GazeDataLoader
from src.gaze_tracking.models.model import GazeTrackingModel
from visualization import plot_training_history, visualize_predictions, plot_gaze_heatmap, display_sample_faces, visualize_data_coverage
from gpu_utils import configure_gpu, get_cuda_info

def train_with_datasets(dataset_paths, args):
    """
    Train a model with one or more datasets.
    """
    all_images = []
    all_gaze_points = []
    
    print(f"Loading data from {len(dataset_paths)} datasets...")
    
    # Loads and combines data from multiple datasets
    for metadata_path in dataset_paths:
        try:
            print(f"\nProcessing dataset: {os.path.basename(os.path.dirname(metadata_path))}")
            # Creates a data loader and loads images and gaze points from the dataset
            data_loader = GazeDataLoader(metadata_path, img_size=(64, 64))
            images, gaze_points = data_loader.load_data()
            
            print(f"Loaded {len(images)} samples from dataset")
            
            # Adds loaded data to the combined dataset
            all_images.append(images)
            all_gaze_points.append(gaze_points)
            
        except Exception as e:
            print(f"Error loading dataset {metadata_path}: {e}")
    
    if not all_images:
        print("No valid data loaded from any dataset.")
        return
    
    # Combines all datasets into single arrays
    images = np.concatenate(all_images, axis=0)
    gaze_points = np.concatenate(all_gaze_points, axis=0)
    
    print(f"Combined dataset: {images.shape[0]} samples")
    
    # Visualizes the dataset to understand its characteristics
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        visualize_data_coverage(gaze_points, 
                              save_path=os.path.join(args.output_dir, 'combined_data_coverage.png'))
        plot_gaze_heatmap(gaze_points, 
                         save_path=os.path.join(args.output_dir, 'combined_gaze_heatmap.png'))
        
        # Only show a sample of face images (max 25)
        sample_size = min(25, len(images))
        sample_indices = np.random.choice(len(images), sample_size, replace=False)
        sample_images = images[sample_indices]
        sample_gaze = gaze_points[sample_indices]
        display_sample_faces(sample_images, sample_gaze, sample_size=sample_size,
                           save_path=os.path.join(args.output_dir, 'sample_faces.png'))
    except Exception as e:
        print(f"Warning: Error during visualization: {e}")
    
    # Splits data into training, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, gaze_points, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Data splits - Train: {X_train.shape[0]}, "
          f"Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Creates the specified model type (resnet50, mobilenet, or custom)
    print(f"Creating {args.model_type} model...")
    model_creator = GazeTrackingModel(
        model_type=args.model_type,
        input_shape=(64, 64, 3),
        use_mixed_precision=args.mixed_precision
    )
    
    # Get model for training
    model = model_creator.model
    model.summary()
    
    # Sets up data augmentation to improve model generalization
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
    )
    datagen.fit(X_train)
    
    # Get callbacks
    callbacks = model_creator.get_callbacks(os.path.join(args.output_dir, 'checkpoints'))
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, X_train.shape[0] // args.batch_size)
    
    try:
        # Trains the model with the augmented data
        print("Starting training...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=args.batch_size),
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=steps_per_epoch
        )
        
        # Plot history
        plot_training_history(history, 
                            save_path=os.path.join(args.output_dir, 'training_history.png'))
        
        # For transfer learning models, performs fine tuning after initial training
        if args.model_type in ['resnet50', 'mobilenet']:
            print("\nFine-tuning by unfreezing top layers...")
            model_creator.unfreeze_top_layers(30)  # Unfreezes top 30 layers
            
            # Additional training with unfrozen layers
            fine_tune_history = model.fit(
                datagen.flow(X_train, y_train, batch_size=args.batch_size),
                validation_data=(X_val, y_val),
                epochs=10,  # Fewer epochs for fine-tuning
                callbacks=callbacks,
                verbose=1,
                steps_per_epoch=steps_per_epoch
            )
            
            # Plot fine-tuning history
            plot_training_history(fine_tune_history, 
                                save_path=os.path.join(args.output_dir, 'fine_tuning_history.png'))
        
        # Evaluate and visualize results
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
        
        # Save final model
        model_save_path = os.path.join(args.output_dir, args.model_path)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Train gaze tracking model")
    parser.add_argument('--dataset_dir', type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output'),
                       help='Directory containing dataset folders')
    parser.add_argument('--metadata_file', type=str, default=None,
                       help='Direct path to a specific metadata file')
    parser.add_argument('--dataset_id', type=str, default=None,
                       help='Specific dataset ID to use (e.g. "00241")')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_type', type=str, default='resnet50', 
                       choices=['resnet50', 'mobilenet', 'custom'], help='Type of model to train')
    parser.add_argument('--model_path', type=str, default="gaze_tracking_model.h5", 
                       help='Path to save the trained model')
    parser.add_argument('--disable_gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--disable_mixed_precision', action='store_true', 
                       help='Disable mixed precision training')
    parser.add_argument('--output_dir', type=str, default="model_output",
                       help='Directory to save model outputs and visualizations')
    parser.add_argument('--list_datasets', action='store_true',
                       help='List available datasets and exit')
    parser.add_argument('--use_all_datasets', action='store_true',
                       help='Use all available datasets for training')
    parser.add_argument('--force_cuda', action='store_true',
                       help='Try harder to find and use CUDA GPU')
    args = parser.parse_args()
    
    # Configure GPU with enhanced detection and reporting
    gpu_available = False if args.disable_gpu else configure_gpu()
    
    # If forcing CUDA and no GPU was found, try setting environment variables
    if args.force_cuda and not gpu_available:
        print("Trying to force CUDA initialization...")
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        gpu_available = configure_gpu()
    
    # Enable mixed precision if GPU is available and not disabled
    args.mixed_precision = False
    if gpu_available and not args.disable_mixed_precision:
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
    
    if args.list_datasets:
        # Already displayed by find_available_datasets
        return
    
    if not available_datasets:
        print(f"No datasets found in {base_dir}")
        return
    
    # Determine which datasets to use
    if args.metadata_file:
        # Single specified metadata file
        selected_datasets = [args.metadata_file]
        print(f"Using specified metadata file: {args.metadata_file}")
    elif args.dataset_id:
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
    
    # Train the model
    train_with_datasets(selected_datasets, args)

if __name__ == "__main__":
    main()
