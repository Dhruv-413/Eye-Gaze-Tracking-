import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard #type: ignore[import]
import matplotlib.pyplot as plt
from model.resnet50 import create_eye_gaze_model
from data.dataset import GazeDataset

def plot_history(history, save_path='training_history.png'):
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

def main(args):
    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = GazeDataset(
        root_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        validation_split=args.val_split
    )
    
    # Create datasets for training and validation
    train_data, val_data = dataset.create_dataset()
    
    # Get the metadata features dimension from the first item
    metadata_features = len(train_data[0]['metadata'])
    
    # Create data generators
    train_generator = dataset.data_generator(train_data, batch_size=args.batch_size)
    val_generator = dataset.data_generator(val_data, batch_size=args.batch_size)
    
    # Calculate steps per epoch
    train_steps = len(train_data) // args.batch_size
    val_steps = len(val_data) // args.batch_size
    
    # Create the model
    model = create_eye_gaze_model(
        input_shape=(args.img_size, args.img_size, 3),
        metadata_shape=metadata_features
    )
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(args.output_dir, 'model_best.h5'),
            monitor='val_mae',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_mae',
            patience=10,
            restore_best_weights=True,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            mode='min'
        ),
        TensorBoard(
            log_dir=os.path.join(args.output_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_history(history, os.path.join(args.output_dir, 'training_history.png'))
    
    # Save the final model
    model.save(os.path.join(args.output_dir, 'model_final.h5'))
    
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a ResNet50-based eye gaze estimation model')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Root directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Directory to save the models and results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Size of input images (square)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    
    args = parser.parse_args()
    main(args)
