import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model #type: ignore[import]
from data.dataset import GazeDataset  # Fix import path

def calculate_angular_error(y_true, y_pred):
    """
    Calculate angular error in degrees between true and predicted gaze vectors
    
    Args:
        y_true: Normalized gaze targets (batch_size, 2)
        y_pred: Predicted gaze points (batch_size, 2)
    
    Returns:
        Angular errors in degrees (batch_size,)
    """
    # Convert normalized screen coordinates to gaze vectors
    # Assume a simplified model where (0,0) is top-left, (1,1) is bottom-right
    # and z-component is fixed (e.g., -1)
    
    # Convert 2D coordinates to 3D gaze vectors (x, y, -1)
    v1 = np.zeros((len(y_true), 3))
    v1[:, 0] = y_true[:, 0] * 2 - 1  # Map from [0,1] to [-1,1]
    v1[:, 1] = -(y_true[:, 1] * 2 - 1)  # Flip y-axis and map from [0,1] to [-1,1]
    v1[:, 2] = -1  # Fixed z component
    
    v2 = np.zeros((len(y_pred), 3))
    v2[:, 0] = y_pred[:, 0] * 2 - 1
    v2[:, 1] = -(y_pred[:, 1] * 2 - 1)
    v2[:, 2] = -1
    
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    
    # Compute cosine similarity
    cos_sim = np.sum(v1 * v2, axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  # Clip to avoid numerical issues
    
    # Convert to angles in degrees
    angles = np.degrees(np.arccos(cos_sim))
    
    return angles

def visualize_predictions(true_points, pred_points, save_path):
    """
    Visualize the true vs. predicted gaze points
    
    Args:
        true_points: True gaze coordinates (N, 2)
        pred_points: Predicted gaze coordinates (N, 2)
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(true_points[:, 0], true_points[:, 1], c='blue', label='Ground Truth', alpha=0.5)
    plt.scatter(pred_points[:, 0], pred_points[:, 1], c='red', label='Predictions', alpha=0.5)
    
    # Draw lines between corresponding points
    for i in range(len(true_points)):
        plt.plot([true_points[i, 0], pred_points[i, 0]], 
                 [true_points[i, 1], pred_points[i, 1]], 
                 'k-', alpha=0.2)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.title('Gaze Estimation Results')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_error_heatmap(true_points, errors, save_path, bins=20):
    """
    Create a heatmap showing errors across the screen
    
    Args:
        true_points: True gaze coordinates (N, 2)
        errors: Angular errors (N,)
        save_path: Path to save the visualization
        bins: Number of bins for the heatmap
    """
    plt.figure(figsize=(12, 10))
    
    # Create a histogram-based heatmap
    heatmap, xedges, yedges = np.histogram2d(
        true_points[:, 0], true_points[:, 1], 
        bins=bins, weights=errors,
        range=[[0, 1], [0, 1]]
    )
    
    # Normalize by count
    counts, _, _ = np.histogram2d(
        true_points[:, 0], true_points[:, 1], 
        bins=bins,
        range=[[0, 1], [0, 1]]
    )
    
    # Avoid division by zero
    mask = counts > 0
    heatmap[mask] = heatmap[mask] / counts[mask]
    
    # Plot heatmap
    plt.imshow(heatmap.T, origin='lower', extent=[0, 1, 0, 1], 
               aspect='auto', cmap='hot_r')
    
    plt.colorbar(label='Mean Angular Error (degrees)')
    plt.title('Error Distribution Across Screen')
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    # Load the trained model
    model = load_model(args.model_path)
    
    # Initialize dataset
    dataset = GazeDataset(
        root_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        validation_split=args.val_split
    )
    
    # Create datasets (we will use the validation set)
    _, val_data = dataset.create_dataset()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on a batch basis to avoid memory issues
    all_predictions = []
    all_targets = []
    
    for batch_idx in range(0, len(val_data), args.batch_size):
        batch_data = val_data[batch_idx:batch_idx + args.batch_size]
        
        # Initialize batch arrays
        batch_size = len(batch_data)
        batch_left_eyes = np.zeros((batch_size, args.img_size, args.img_size, 3))
        batch_right_eyes = np.zeros((batch_size, args.img_size, args.img_size, 3))
        batch_faces = np.zeros((batch_size, args.img_size, args.img_size, 3))
        batch_metadata = np.zeros((batch_size, len(batch_data[0]['metadata'])))
        batch_targets = np.zeros((batch_size, 2))
        
        # Fill the batch
        for i, sample in enumerate(batch_data):
            batch_left_eyes[i] = dataset.preprocess_image(sample['left_eye_path'])
            batch_right_eyes[i] = dataset.preprocess_image(sample['right_eye_path'])
            batch_faces[i] = dataset.preprocess_image(sample['face_path'])
            batch_metadata[i] = sample['metadata']
            batch_targets[i] = sample['gaze_target']
        
        # Get predictions with the correct input structure
        batch_predictions = model.predict({
            'left_eye_input': batch_left_eyes,
            'right_eye_input': batch_right_eyes,
            'face_input': batch_faces,
            'metadata_input': batch_metadata
        })
        
        all_predictions.extend(batch_predictions)
        all_targets.extend(batch_targets)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_predictions - all_targets))
    mse = np.mean(np.square(all_predictions - all_targets))
    angular_errors = calculate_angular_error(all_targets, all_predictions)
    mean_angular_error = np.mean(angular_errors)
    median_angular_error = np.median(angular_errors)
    
    # Print results
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Angular Error: {mean_angular_error:.2f} degrees")
    print(f"Median Angular Error: {median_angular_error:.2f} degrees")
    
    # Save visualization of predictions
    visualize_predictions(
        all_targets, all_predictions,
        os.path.join(args.output_dir, 'prediction_visualization.png')
    )
    
    # Create error heatmap
    create_error_heatmap(
        all_targets, angular_errors,
        os.path.join(args.output_dir, 'error_heatmap.png')
    )
    
    # Create histogram of angular errors
    plt.figure(figsize=(10, 6))
    plt.hist(angular_errors, bins=50, alpha=0.75)
    plt.axvline(mean_angular_error, color='r', linestyle='--', 
                label=f'Mean: {mean_angular_error:.2f}°')
    plt.axvline(median_angular_error, color='g', linestyle='--', 
                label=f'Median: {median_angular_error:.2f}°')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Angular Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'error_histogram.png'))
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Mean Absolute Error: {mae:.4f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write(f"Mean Angular Error: {mean_angular_error:.2f} degrees\n")
        f.write(f"Median Angular Error: {median_angular_error:.2f} degrees\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained eye gaze estimation model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.h5 file)')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Root directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Size of input images (square)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    
    args = parser.parse_args()
    main(args)
