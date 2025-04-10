import os
import argparse
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model #type: ignore[import]
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type: ignore[import]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_metadata_features(metadata):
    """Extract relevant features from metadata entry"""
    features = []
    
    # Add screen info
    if "screen_data" in metadata:
        features.append(metadata["screen_data"]["W"] / 1000.0)  # Normalize width
        features.append(metadata["screen_data"]["H"] / 1000.0)  # Normalize height
        features.append(metadata["screen_data"]["Orientation"] / 4.0)  # Normalize orientation
    else:
        features.extend([0.0, 0.0, 0.0])
        
    # Add face grid data if available
    if "face_grid_data" in metadata and isinstance(metadata["face_grid_data"], dict):
        if "X" in metadata["face_grid_data"]:
            features.append(metadata["face_grid_data"]["X"] / 25.0)  # Normalize
        else:
            features.append(0.0)
            
        if "Y" in metadata["face_grid_data"]:
            features.append(metadata["face_grid_data"]["Y"] / 25.0)  # Normalize
        else:
            features.append(0.0)
            
        if "W" in metadata["face_grid_data"]:
            features.append(metadata["face_grid_data"]["W"] / 25.0)  # Normalize
        else:
            features.append(0.0)
            
        if "H" in metadata["face_grid_data"]:
            features.append(metadata["face_grid_data"]["H"] / 25.0)  # Normalize
        else:
            features.append(0.0)
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
        
    # Add some motion data if available
    if "motion_data" in metadata and isinstance(metadata["motion_data"], dict):
        if "Time" in metadata["motion_data"]:
            features.append(metadata["motion_data"]["Time"])  # Already normalized
        else:
            features.append(0.0)
        
        # Add DotNum as a feature
        if "DotNum" in metadata["motion_data"]:
            features.append(metadata["motion_data"]["DotNum"] / 23.0)  # Normalize (max is 23)
        else:
            features.append(0.0)
    else:
        features.extend([0.0, 0.0])
        
    return np.array(features, dtype=np.float32).reshape(1, -1)

def visualize_prediction(image_path, gaze_coords, output_path, model_type=""):
    """Visualize the predicted gaze point on the face image"""
    # Read the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    x, y = int(gaze_coords[0] * w), int(gaze_coords[1] * h)
    
    # Draw a crosshair on the gaze point
    cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    
    # Draw a circle around the gaze point
    cv2.circle(image, (x, y), 10, (0, 255, 0), 2)
    
    # Add text with coordinates
    text = f"Gaze: ({gaze_coords[0]:.2f}, {gaze_coords[1]:.2f})"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    # Add model type if provided
    if model_type:
        cv2.putText(image, f"Model: {model_type}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    
    # Save the visualization
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to {output_path}")

def main(args):
    # Load the model
    model = load_model(args.model_path)
    
    # Load and preprocess the images
    left_eye = preprocess_image(args.left_eye, (args.img_size, args.img_size))
    right_eye = preprocess_image(args.right_eye, (args.img_size, args.img_size))
    face = preprocess_image(args.face, (args.img_size, args.img_size))
    
    # Load metadata if provided
    metadata_features = None
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
        metadata_features = extract_metadata_features(metadata)
    else:
        # Create default metadata with correct shape
        metadata_features = np.zeros((1, 10), dtype=np.float32)
    
    # Make prediction with proper input dictionary structure
    prediction = model.predict({
        'left_eye_input': left_eye,
        'right_eye_input': right_eye,
        'face_input': face,
        'metadata_input': metadata_features
    })
    
    # Handle different model outputs - if it's a multitask model, 
    # prediction will be a list where first element is gaze predictions
    if isinstance(prediction, list):
        gaze_coords = prediction[0][0]
        # If the model includes pose estimation, we can also display it
        if len(prediction) > 1:
            pose_angles = prediction[1][0]
            print(f"Predicted head pose (pitch, yaw, roll): [{pose_angles[0]:.2f}, {pose_angles[1]:.2f}, {pose_angles[2]:.2f}]")
    else:
        gaze_coords = prediction[0]
    
    print(f"Predicted gaze coordinates (normalized): [{gaze_coords[0]:.4f}, {gaze_coords[1]:.4f}]")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Determine the model type from the filename
    model_type = ""
    if "hybrid" in args.model_path.lower():
        model_type = "Hybrid ResNet-GazeNet"
    elif "resnet" in args.model_path.lower():
        model_type = "ResNet50"
    elif "efficient" in args.model_path.lower() or "gazenet" in args.model_path.lower():
        model_type = "EfficientNet/GazeNet"
    
    # Visualize the prediction on the face image
    visualize_prediction(args.face, gaze_coords, args.output, model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict gaze from images using a trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.h5 file)')
    parser.add_argument('--left_eye', type=str, required=True,
                        help='Path to the left eye image')
    parser.add_argument('--right_eye', type=str, required=True,
                        help='Path to the right eye image')
    parser.add_argument('--face', type=str, required=True,
                        help='Path to the face image')
    parser.add_argument('--metadata', type=str,
                        help='Path to JSON file with metadata')
    parser.add_argument('--output', type=str, default='gaze_prediction.jpg',
                        help='Path to save the visualization')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Size of input images (square)')
    
    args = parser.parse_args()
    main(args)
