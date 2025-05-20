import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard #type: ignore[import]
from datetime import datetime
from model.hybrid_model import create_hybrid_model, get_hybrid_model_callbacks
from data.dataset import GazeDataset
import matplotlib.pyplot as plt

def train_hybrid_model(
    data_dir,
    trained_model,
    batch_size=32,
    epochs=50,
    learning_rate=0.0005,
    image_size=224,
    use_mixed_precision=True,
    include_pose_estimation=True,
    fusion_method='feature',
    val_split=0.2
):
    """
    Train the hybrid model that combines ResNet50 and GazeNet features
    
    Args:
        data_dir: Directory containing the dataset
        trained_model: Directory to save trained model and logs
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        image_size: Size of input images
        use_mixed_precision: Whether to use mixed precision training
        include_pose_estimation: Whether to include pose estimation task
        fusion_method: Method for fusing features ('feature' or 'decision')
        val_split: Fraction of data to use for validation
    """
    # Create output directory if it doesn't exist
    os.makedirs(trained_model, exist_ok=True)
    
    # Create model checkpoint path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(trained_model, f"hybrid_model_{timestamp}.h5")
    
    # Create TensorBoard log directory
    log_dir = os.path.join(trained_model, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize dataset
    print(f"Initializing dataset from {data_dir}...")
    dataset = GazeDataset(
        root_dir=data_dir,
        img_size=(image_size, image_size),
        validation_split=val_split
    )
    
    # Create datasets for training and validation
    train_data, val_data = dataset.create_dataset()
    
    # Get the metadata features dimension from the first item
    metadata_features = len(train_data[0]['metadata']) if train_data else 10
    
    # Create data generators
    train_generator = dataset.data_generator(train_data, batch_size=batch_size, include_pose=include_pose_estimation)
    val_generator = dataset.data_generator(val_data, batch_size=batch_size, include_pose=include_pose_estimation)
    
    # Calculate steps per epoch
    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size
    
    # Create the hybrid model
    print(f"Creating hybrid model with fusion method: {fusion_method}")
    model = create_hybrid_model(
        input_shape=(image_size, image_size, 3),
        metadata_shape=metadata_features,
        learning_rate=learning_rate,
        use_mixed_precision=use_mixed_precision,
        include_pose_estimation=include_pose_estimation,
        fusion_method=fusion_method
    )
    
    # Print model summary
    model.summary()
    
    # Get callbacks
    callbacks = get_hybrid_model_callbacks(checkpoint_path)
    
    # Add TensorBoard callback
    callbacks.append(TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    ))
    
    # Train the model
    print(f"Starting training for {epochs} epochs...")
    # if include_pose_estimation:
        # When using pose estimation, the generator needs to provide pose data
        # This is a placeholder - you'd need to modify your data generator
        # to provide the necessary pose information
        
        # Placeholder for pose data training
        # In a real implementation, your generator would yield a tuple:
        # ((inputs), {'gaze_output': gaze_targets, 'pose_output': pose_targets})
        
        # For now, let's simply train without the pose estimation
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(trained_model, f"hybrid_model_final_{timestamp}.h5")
    model.save(final_model_path)
    print(f"Model training complete. Final model saved to {final_model_path}")
    
    # Plot training history
    plot_history(history, os.path.join(trained_model, f"hybrid_training_history_{timestamp}.png"))

def plot_history(history, save_path):
    """Plot and save the training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid model for eye gaze estimation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--trained_model", type=str, default="./output/hybrid_model", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--no_pose", action="store_true", help="Disable pose estimation")
    parser.add_argument("--fusion", type=str, default="feature", choices=["feature", "decision"], 
                        help="Feature fusion method (feature or decision)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    train_hybrid_model(
        data_dir=args.data_dir,
        trained_model=args.trained_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        image_size=args.image_size,
        use_mixed_precision=args.mixed_precision,
        include_pose_estimation=not args.no_pose,
        fusion_method=args.fusion,
        val_split=args.val_split
    )
