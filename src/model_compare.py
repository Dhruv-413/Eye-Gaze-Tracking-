import os
import argparse
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore[import]
from data.dataset import GazeDataset

def evaluate_model(model_path, dataset, batch_size=32):
    """
    Evaluate a model on the validation set
    
    Args:
        model_path: Path to the trained model
        dataset: GazeDataset instance
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load the model
    model = load_model(model_path)
    
    # Create datasets for validation
    _, val_data = dataset.create_dataset()
    
    # Create batches for prediction
    all_predictions = []
    all_targets = []
    inference_times = []
    
    for batch_idx in range(0, len(val_data), batch_size):
        batch_data = val_data[batch_idx:batch_idx + batch_size]
        
        # Initialize batch arrays
        batch_size = len(batch_data)
        batch_left_eyes = np.zeros((batch_size, dataset.img_size[0], dataset.img_size[1], 3))
        batch_right_eyes = np.zeros((batch_size, dataset.img_size[0], dataset.img_size[1], 3))
        batch_faces = np.zeros((batch_size, dataset.img_size[0], dataset.img_size[1], 3))
        batch_metadata = np.zeros((batch_size, len(batch_data[0]['metadata'])))
        batch_targets = np.zeros((batch_size, 2))
        
        # Fill the batch
        for i, sample in enumerate(batch_data):
            batch_left_eyes[i] = dataset.preprocess_image(sample['left_eye_path'])
            batch_right_eyes[i] = dataset.preprocess_image(sample['right_eye_path'])
            batch_faces[i] = dataset.preprocess_image(sample['face_path'])
            batch_metadata[i] = sample['metadata']
            batch_targets[i] = sample['gaze_target']
        
        # Measure inference time
        start_time = time.time()
        batch_predictions = model.predict({
            'left_eye_input': batch_left_eyes,
            'right_eye_input': batch_right_eyes,
            'face_input': batch_faces,
            'metadata_input': batch_metadata
        }, verbose=0)
        end_time = time.time()
        
        # Handle different model outputs - if it's a multitask model, 
        # batch_predictions will be a list where first element is gaze predictions
        if isinstance(batch_predictions, list):
            batch_predictions = batch_predictions[0]  # Get gaze predictions only
        
        inference_times.append(end_time - start_time)
        all_predictions.extend(batch_predictions)
        all_targets.extend(batch_targets)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_predictions - all_targets))
    mse = np.mean(np.square(all_predictions - all_targets))
    
    # Calculate angular error (simplified for this example)
    def calculate_angular_error(y_true, y_pred):
        # Convert normalized screen coordinates to gaze vectors
        v1 = np.zeros((len(y_true), 3))
        v1[:, 0] = y_true[:, 0] * 2 - 1  # Map from [0,1] to [-1,1]
        v1[:, 1] = -(y_true[:, 1] * 2 - 1)
        v1[:, 2] = -1
        
        v2 = np.zeros((len(y_pred), 3))
        v2[:, 0] = y_pred[:, 0] * 2 - 1
        v2[:, 1] = -(y_pred[:, 1] * 2 - 1)
        v2[:, 2] = -1
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        
        # Compute cosine similarity
        cos_sim = np.sum(v1 * v2, axis=1)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # Convert to angles in degrees
        return np.degrees(np.arccos(cos_sim))
    
    angular_errors = calculate_angular_error(all_targets, all_predictions)
    mean_angular_error = np.mean(angular_errors)
    
    # Get model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Get average inference time
    avg_inference_time = np.mean(inference_times[1:])  # Skip the first prediction as it includes compilation
    avg_inference_time_per_sample = avg_inference_time / batch_size
    
    return {
        'mae': mae,
        'mse': mse,
        'mean_angular_error': mean_angular_error,
        'model_size_mb': model_size_mb,
        'avg_inference_time': avg_inference_time,
        'avg_inference_time_per_sample': avg_inference_time_per_sample,
        'num_samples': len(all_targets)
    }

def plot_comparison(results, save_path):
    """
    Plot comparison between different models
    
    Args:
        results: Dictionary of model results
        save_path: Path to save the comparison plot
    """
    models = list(results.keys())
    metrics = ['mae', 'mse', 'mean_angular_error', 'model_size_mb', 'avg_inference_time_per_sample']
    metric_names = ['MAE', 'MSE', 'Mean Angular Error (°)', 'Model Size (MB)', 'Inference Time per Sample (s)']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[model][metric] for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(name)
        axes[i].set_ylabel(name)
        
        # Add values on top of bars
        for j, v in enumerate(values):
            axes[i].text(j, v, f"{v:.4f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    # Initialize dataset
    dataset = GazeDataset(
        root_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        validation_split=args.val_split
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Evaluate ResNet50 model if provided
    if args.resnet_model:
        print("\nEvaluating ResNet50 model...")
        results['ResNet50'] = evaluate_model(args.resnet_model, dataset, args.batch_size)
        print(f"ResNet50 Results:")
        for k, v in results['ResNet50'].items():
            print(f"  {k}: {v}")
            
    # Evaluate EfficientNet model if provided
    if args.efficientnet_model:
        print("\nEvaluating EfficientNet model...")
        results['EfficientNet'] = evaluate_model(args.efficientnet_model, dataset, args.batch_size)
        print(f"EfficientNet Results:")
        for k, v in results['EfficientNet'].items():
            print(f"  {k}: {v}")
    
    # Evaluate Hybrid model if provided
    if args.hybrid_model:
        print("\nEvaluating Hybrid ResNet-GazeNet model...")
        results['Hybrid'] = evaluate_model(args.hybrid_model, dataset, args.batch_size)
        print(f"Hybrid Model Results:")
        for k, v in results['Hybrid'].items():
            print(f"  {k}: {v}")
    
    # Plot comparison if multiple models are provided
    model_count = sum(1 for x in [args.resnet_model, args.efficientnet_model, args.hybrid_model] if x)
    if model_count > 1:
        plot_comparison(results, os.path.join(args.output_dir, 'model_comparison.png'))
    
    # Save results to text file
    with open(os.path.join(args.output_dir, 'comparison_results.txt'), 'w') as f:
        for model_name, model_results in results.items():
            f.write(f"{model_name} Results:\n")
            for k, v in model_results.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare models for eye gaze estimation')
    parser.add_argument('--resnet_model', type=str, 
                        help='Path to trained ResNet50 model (.h5 file)')
    parser.add_argument('--efficientnet_model', type=str, 
                        help='Path to trained EfficientNet model (.h5 file)')
    parser.add_argument('--hybrid_model', type=str, 
                        help='Path to trained Hybrid model (.h5 file)')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Root directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./model_comparison',
                        help='Directory to save comparison results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Size of input images (square)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    
    args = parser.parse_args()
    
    if not any([args.resnet_model, args.efficientnet_model, args.hybrid_model]):
        raise ValueError("At least one model path must be provided")
    
    main(args)
