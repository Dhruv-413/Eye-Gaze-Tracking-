"""
Module for visualizing gaze data and model performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from typing import Dict, Any, Optional, Tuple, List
from mpl_toolkits.mplot3d import Axes3D

def plot_training_history(history, save_path: Optional[str] = None):
    """
    Visualize model training history.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    # Creates a figure with two subplots
    plt.figure(figsize=(12, 4))
    
    # First subplot shows the training and validation loss over epochs
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Second subplot shows Mean Absolute Error metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Saves the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def visualize_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                          save_path: Optional[str] = None,
                          show_error: bool = True):
    """
    Visualize prediction results.
    
    Args:
        y_true: Ground truth gaze points
        y_pred: Predicted gaze points
        save_path: Path to save the plot
        show_error: Whether to visualize prediction errors with arrows
    """
    # Creates scatter plot comparing true vs predicted gaze points
    plt.figure(figsize=(10, 10))
    
    # Plots actual gaze points in blue and predicted in red
    plt.scatter(y_true[:, 0], y_true[:, 1], c='blue', label='Actual', alpha=0.7, s=30)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Predicted', alpha=0.7, s=30)
    
    # Draws a dashed line representing screen boundaries
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', alpha=0.5)
    
    # Draws green arrows showing prediction errors for each point (limited to 100 points)
    if show_error and len(y_true) <= 100:
        for i in range(len(y_true)):
            plt.arrow(y_true[i, 0], y_true[i, 1], 
                     y_pred[i, 0] - y_true[i, 0], 
                     y_pred[i, 1] - y_true[i, 1], 
                     color='green', alpha=0.3, width=0.001)
    
    # Calculates and displays the average Euclidean error
    avg_error = np.mean(np.sqrt(np.sum((y_true - y_pred)**2, axis=1)))
    plt.title(f'Gaze Prediction Results\nAverage Error: {avg_error:.4f}')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Prediction visualization saved to {save_path}")
    
    plt.show()

def plot_gaze_heatmap(gaze_points: np.ndarray, save_path: Optional[str] = None):
    """
    Create a heatmap from gaze points.
    
    Args:
        gaze_points: Array of gaze coordinates (x, y)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    plt.hist2d(gaze_points[:, 0], gaze_points[:, 1], bins=50, cmap='hot')
    plt.colorbar(label='Point density')
    
    # Draw screen boundary
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'w--', alpha=0.7)
    
    plt.title(f'Gaze Point Distribution Heatmap ({len(gaze_points)} points)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
    
    plt.show()

def display_sample_faces(images: np.ndarray, gaze_points: np.ndarray, 
                        sample_size: int = 10, save_path: Optional[str] = None):
    """
    Display a grid of sample face images with their corresponding gaze points.
    
    Args:
        images: Face images (normalized)
        gaze_points: Corresponding gaze points
        sample_size: Number of sample images to show
        save_path: Path to save the plot
    """
    # Limit sample size to available images
    sample_size = min(sample_size, len(images))
    indices = np.random.choice(len(images), sample_size, replace=False)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(sample_size)))
    
    plt.figure(figsize=(15, 15))
    
    for i, idx in enumerate(indices):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Display the image (denormalize from [0,1] to [0,255])
        img = images[idx] * 255
        img = img.astype(np.uint8)
        
        # Handle both RGB and grayscale images
        if img.shape[-1] == 1:
            plt.imshow(img[:, :, 0], cmap='gray')
        else:
            plt.imshow(img)
        
        x, y = gaze_points[idx]
        plt.title(f'Gaze: ({x:.2f}, {y:.2f})')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Sample faces saved to {save_path}")
    
    plt.show()

def visualize_data_coverage(gaze_points: np.ndarray, save_path: Optional[str] = None):
    """
    Visualize the coverage of gaze points across the screen space.
    
    Args:
        gaze_points: Array of gaze coordinates (x, y)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create main scatter plot
    plt.scatter(gaze_points[:, 0], gaze_points[:, 1], alpha=0.5, s=10)
    
    # Draw screen boundary
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'r--', alpha=0.7, 
             label='Screen boundary')
    
    # Add grid and division lines
    plt.grid(True, alpha=0.3)
    for i in range(1, 3):
        plt.axhline(y=i/3, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=i/3, color='gray', linestyle='--', alpha=0.3)
    
    # Count points in each area to analyze distribution
    x_bins = np.linspace(0, 1, 4)
    y_bins = np.linspace(0, 1, 4)
    
    h, _, _ = np.histogram2d(gaze_points[:, 0], gaze_points[:, 1], 
                           bins=[x_bins, y_bins])
    
    total_points = len(gaze_points)
    coverage_percentage = np.count_nonzero(h) / 9 * 100  # 9 areas
    
    plt.title(f'Gaze Point Coverage ({len(gaze_points)} points)\n'
              f'Grid Coverage: {coverage_percentage:.1f}% of screen regions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Coverage visualization saved to {save_path}")
    
    plt.show()

def visualize_head_pose(image: np.ndarray, euler_angles: Tuple[float, float, float], 
                       save_path: Optional[str] = None):
    """
    Visualize head pose angles in 3D.
    
    Args:
        image: Input image (for thumbnail)
        euler_angles: (pitch, yaw, roll) angles in degrees
        save_path: Path to save the visualization
    """
    pitch, yaw, roll = euler_angles
    
    # Create figure with 3D plot
    fig = plt.figure(figsize=(10, 5))
    
    # Add image on the left
    ax1 = fig.add_subplot(121)
    if image.shape[-1] == 3:
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(image, cmap='gray')
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # 3D visualization on the right
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Draw a basic head model (just a cuboid for simplicity)
    # Define vertices of a cuboid centered at origin
    size = 1.0
    vertices = np.array([
        [-size, -size, -size],  # 0
        [size, -size, -size],   # 1
        [size, size, -size],    # 2
        [-size, size, -size],   # 3
        [-size, -size, size],   # 4
        [size, -size, size],    # 5
        [size, size, size],     # 6
        [-size, size, size]     # 7
    ])
    
    # Define edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    # Convert angles from degrees to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)
    
    # Create rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # Apply rotation to vertices
    rotated_vertices = np.dot(vertices, R.T)
    
    # Draw edges
    for edge in edges:
        ax2.plot3D(
            [rotated_vertices[edge[0], 0], rotated_vertices[edge[1], 0]],
            [rotated_vertices[edge[0], 1], rotated_vertices[edge[1], 1]],
            [rotated_vertices[edge[0], 2], rotated_vertices[edge[1], 2]],
            'b'
        )
    
    # Draw coordinate axes
    axis_length = 2.0
    origin = np.array([0, 0, 0])
    
    # Original axes
    ax2.quiver(origin[0], origin[1], origin[2], 
               axis_length, 0, 0, color='r', arrow_length_ratio=0.1)
    ax2.quiver(origin[0], origin[1], origin[2], 
               0, axis_length, 0, color='g', arrow_length_ratio=0.1)
    ax2.quiver(origin[0], origin[1], origin[2], 
               0, 0, axis_length, color='b', arrow_length_ratio=0.1)
    
    # Add text labels for axes
    ax2.text(axis_length, 0, 0, "X", color='red')
    ax2.text(0, axis_length, 0, "Y", color='green')
    ax2.text(0, 0, axis_length, "Z", color='blue')
    
    # Add text with angle values
    ax2.text2D(0.05, 0.95, f"Pitch: {pitch:.1f}°", transform=ax2.transAxes)
    ax2.text2D(0.05, 0.90, f"Yaw: {yaw:.1f}°", transform=ax2.transAxes)
    ax2.text2D(0.05, 0.85, f"Roll: {roll:.1f}°", transform=ax2.transAxes)
    
    # Set equal aspect ratio
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim([-axis_length, axis_length])
    ax2.set_ylim([-axis_length, axis_length])
    ax2.set_zlim([-axis_length, axis_length])
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Head Pose Visualization')
    
    plt.tight_layout()
    
    # Save visualization if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Head pose visualization saved to {save_path}")
    
    plt.show()

def plot_head_pose_distribution(poses: List[Tuple[float, float, float]], 
                               save_path: Optional[str] = None):
    """
    Visualize the distribution of head poses in the dataset.
    
    Args:
        poses: List of (pitch, yaw, roll) tuples
        save_path: Path to save the plot
    """
    if not poses:
        print("No head pose data to visualize")
        return
        
    # Convert list of tuples to array
    poses_array = np.array(poses)
    pitch = poses_array[:, 0]
    yaw = poses_array[:, 1]
    roll = poses_array[:, 2]
    
    plt.figure(figsize=(15, 5))
    
    # Plot pitch distribution
    plt.subplot(131)
    plt.hist(pitch, bins=30, color='red', alpha=0.7)
    plt.title(f'Pitch Distribution\nMean: {np.mean(pitch):.1f}°, Std: {np.std(pitch):.1f}°')
    plt.xlabel('Pitch (degrees)')
    plt.ylabel('Frequency')
    
    # Plot yaw distribution
    plt.subplot(132)
    plt.hist(yaw, bins=30, color='green', alpha=0.7)
    plt.title(f'Yaw Distribution\nMean: {np.mean(yaw):.1f}°, Std: {np.std(yaw):.1f}°')
    plt.xlabel('Yaw (degrees)')
    
    # Plot roll distribution
    plt.subplot(133)
    plt.hist(roll, bins=30, color='blue', alpha=0.7)
    plt.title(f'Roll Distribution\nMean: {np.mean(roll):.1f}°, Std: {np.std(roll):.1f}°')
    plt.xlabel('Roll (degrees)')
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Head pose distribution plot saved to {save_path}")
    
    plt.show()
    
    # Create 2D scatter plots for angle combinations
    plt.figure(figsize=(15, 5))
    
    # Plot pitch vs yaw
    plt.subplot(131)
    plt.scatter(yaw, pitch, alpha=0.5, s=10)
    plt.title('Pitch vs Yaw')
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Pitch (degrees)')
    plt.grid(True, alpha=0.3)
    
    # Plot pitch vs roll
    plt.subplot(132)
    plt.scatter(roll, pitch, alpha=0.5, s=10)
    plt.title('Pitch vs Roll')
    plt.xlabel('Roll (degrees)')
    plt.ylabel('Pitch (degrees)')
    plt.grid(True, alpha=0.3)
    
    # Plot yaw vs roll
    plt.subplot(133)
    plt.scatter(roll, yaw, alpha=0.5, s=10)
    plt.title('Yaw vs Roll')
    plt.xlabel('Roll (degrees)')
    plt.ylabel('Yaw (degrees)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        base, ext = os.path.splitext(save_path)
        scatter_path = f"{base}_scatter{ext}"
        plt.savefig(scatter_path, dpi=300)
        print(f"Head pose scatter plot saved to {scatter_path}")
    
    plt.show()
