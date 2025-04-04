#!/usr/bin/env python3
"""
Inference script for gaze tracking on static images or webcam feed.
This script loads a trained gaze tracking model and performs inference on images or in real-time.
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Union

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.gaze_tracking.visualization import visualize_predictions, visualize_head_pose

def load_model(model_path: str) -> Tuple[tf.keras.Model, str]:
    """
    Load a trained gaze tracking model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (model, model_type)
    """
    print(f"Loading model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        model.summary()
        
        # Infer model type from input structure
        model_type = "head_pose"
        
        if isinstance(model.input, dict) or isinstance(model.input, list):
            input_names = [layer.name for layer in model.inputs]
            
            if any('head_pose' in name for name in input_names):
                model_type = "head_pose"
            elif any('eye' in name for name in input_names):
                model_type = "multi_input"
                
        print(f"Detected model type: {model_type}")
        return model, model_type
    
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

def preprocess_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess an image for the model.
    
    Args:
        img: Input image
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image
    """
    # Resize
    img_resized = cv2.resize(img, target_size)
    
    # Convert to RGB if it's a BGR image
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_resized = img_resized.astype(np.float32) / 255.0
    
    return img_resized

def setup_mediapipe() -> Tuple[mp.solutions.face_mesh.FaceMesh, Dict]:
    """
    Set up MediaPipe face mesh for face and eye detection.
    
    Returns:
        Tuple of (face_mesh, landmark_indices)
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Define key landmark indices
    landmark_indices = {
        # Left eye landmarks
        'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        # Right eye landmarks
        'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        # Face landmarks for pose estimation
        'pose': [1, 33, 61, 199, 263, 291]  # Nose, eyes, mouth corners
    }
    
    return face_mesh, landmark_indices

def estimate_head_pose(landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Estimate head pose (Euler angles) from face landmarks.
    
    Args:
        landmarks: Face landmarks from MediaPipe
        frame_shape: Frame dimensions (height, width)
        
    Returns:
        Tuple of (pitch, yaw, roll) in degrees
    """
    # 3D model points (standard face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ]) / 4.5  # Scale down for better visualization
    
    # Camera internals
    h, w = frame_shape
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))
    
    # Get specific landmarks for pose estimation
    image_points = np.array([
        landmarks[1],    # Nose tip
        landmarks[152],  # Chin
        landmarks[226],  # Left eye left corner
        landmarks[446],  # Right eye right corner
        landmarks[57],   # Left mouth corner
        landmarks[287]   # Right mouth corner
    ])[:, :2]  # Keep only x,y
    
    # Solve for pose
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        return 0.0, 0.0, 0.0
    
    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    
    # Convert to Euler angles
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    
    pitch, yaw, roll = euler_angles.flatten()
    
    # Convert to degrees and adjust signs
    pitch = -pitch
    yaw = -yaw
    
    return pitch, yaw, roll

def process_image(
    img_path: str,
    model: tf.keras.Model,
    model_type: str,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
    landmark_indices: Dict,
    target_size: Tuple[int, int] = (64, 64)
) -> Dict:
    """
    Process an image for gaze prediction.
    
    Args:
        img_path: Path to the input image
        model: Trained model
        model_type: Type of model ('standard', 'multi_input', or 'head_pose')
        face_mesh: MediaPipe face mesh
        landmark_indices: Dictionary of landmark indices
        target_size: Target size for image preprocessing
        
    Returns:
        Dictionary with processing results
    """
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        return {"success": False, "error": f"Could not read image at {img_path}"}
    
    orig_img = img.copy()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = face_mesh.process(rgb_img)
    
    if not results.multi_face_landmarks:
        return {"success": False, "error": "No face detected"}
    
    # Get face landmarks
    h, w = img.shape[:2]
    landmarks = np.array([
        [lm.x * w, lm.y * h, lm.z * w]
        for lm in results.multi_face_landmarks[0].landmark
    ])
    
    # Extract face region
    face_min = np.min(landmarks[:, :2], axis=0).astype(int)
    face_max = np.max(landmarks[:, :2], axis=0).astype(int)
    
    # Add margin (10% of face size)
    face_size = face_max - face_min
    margin = (face_size * 0.1).astype(int)
    face_min = np.maximum(0, face_min - margin)
    face_max = np.minimum([w, h], face_max + margin)
    
    face_img = img[face_min[1]:face_max[1], face_min[0]:face_max[0]]
    
    # Prepare inputs based on model type
    if model_type == "standard":
        # Standard model only needs the face image
        face_input = preprocess_image(face_img, target_size)
        face_input = np.expand_dims(face_input, axis=0)  # Add batch dimension
        
        # Predict
        prediction = model.predict(face_input, verbose=0)[0]
        
    else:
        # For multi-input or head_pose models, we need more data
        
        # Extract eye regions
        left_eye_pts = landmarks[landmark_indices['left_eye']]
        right_eye_pts = landmarks[landmark_indices['right_eye']]
        
        # Calculate eye bounding boxes
        left_eye_min = np.min(left_eye_pts[:, :2], axis=0).astype(int)
        left_eye_max = np.max(left_eye_pts[:, :2], axis=0).astype(int)
        right_eye_min = np.min(right_eye_pts[:, :2], axis=0).astype(int)
        right_eye_max = np.max(right_eye_pts[:, :2], axis=0).astype(int)
        
        # Add margin
        eye_margin = 5
        left_eye_min = np.maximum(0, left_eye_min - eye_margin)
        left_eye_max = np.minimum([w, h], left_eye_max + eye_margin)
        right_eye_min = np.maximum(0, right_eye_min - eye_margin)
        right_eye_max = np.minimum([w, h], right_eye_max + eye_margin)
        
        # Extract eye images
        left_eye_img = img[left_eye_min[1]:left_eye_max[1], left_eye_min[0]:left_eye_max[0]]
        right_eye_img = img[right_eye_min[1]:right_eye_max[1], right_eye_min[0]:right_eye_max[0]]
        
        if model_type == "multi_input":
            # Preprocess all images
            face_input = preprocess_image(face_img, target_size)
            left_eye_input = preprocess_image(left_eye_img, target_size)
            right_eye_input = preprocess_image(right_eye_img, target_size)
            
            # Add batch dimension
            face_input = np.expand_dims(face_input, axis=0)
            left_eye_input = np.expand_dims(left_eye_input, axis=0)
            right_eye_input = np.expand_dims(right_eye_input, axis=0)
            
            # Prepare model inputs
            model_inputs = {
                'face_input': face_input,
                'left_eye_input': left_eye_input,
                'right_eye_input': right_eye_input
            }
            
            # Predict
            prediction = model.predict(model_inputs, verbose=0)[0]
            
        elif model_type == "head_pose":
            # Estimate head pose
            pitch, yaw, roll = estimate_head_pose(landmarks, (h, w))
            
            # Preprocess face image
            face_input = preprocess_image(face_img, target_size)
            face_input = np.expand_dims(face_input, axis=0)
            
            # Prepare head pose input
            head_pose_input = np.array([[pitch, yaw, roll]], dtype=np.float32)
            
            # Prepare model inputs
            model_inputs = {
                'face_input': face_input,
                'head_pose_input': head_pose_input
            }
            
            # Predict
            prediction = model.predict(model_inputs, verbose=0)[0]
    
    # Clip prediction to [0, 1] range
    prediction = np.clip(prediction, 0, 1)
    
    # Prepare result
    result = {
        "success": True,
        "image": orig_img,
        "face": face_img,
        "prediction": prediction,
        "face_box": (face_min[0], face_min[1], face_max[0], face_max[1])
    }
    
    # Add extra data based on model type
    if model_type == "multi_input":
        result["left_eye"] = left_eye_img
        result["right_eye"] = right_eye_img
        
    elif model_type == "head_pose":
        result["head_pose"] = (pitch, yaw, roll)
    
    return result

def process_frame(
    frame: np.ndarray,
    model: tf.keras.Model,
    model_type: str,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
    landmark_indices: Dict,
    target_size: Tuple[int, int] = (64, 64)
) -> Dict:
    """
    Process a frame for gaze prediction (real-time version).
    
    Args:
        frame: Input frame from camera
        model: Trained model
        model_type: Type of model ('standard', 'multi_input', or 'head_pose')
        face_mesh: MediaPipe face mesh
        landmark_indices: Dictionary of landmark indices
        target_size: Target size for image preprocessing
        
    Returns:
        Dictionary with processing results
    """
    # Make a copy of the frame
    orig_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return {"success": False, "error": "No face detected"}
    
    # Get face landmarks
    h, w = frame.shape[:2]
    landmarks = np.array([
        [lm.x * w, lm.y * h, lm.z * w]
        for lm in results.multi_face_landmarks[0].landmark
    ])
    
    # Extract face region
    face_min = np.min(landmarks[:, :2], axis=0).astype(int)
    face_max = np.max(landmarks[:, :2], axis=0).astype(int)
    
    # Add margin (10% of face size)
    face_size = face_max - face_min
    margin = (face_size * 0.1).astype(int)
    face_min = np.maximum(0, face_min - margin)
    face_max = np.minimum([w, h], face_max + margin)
    
    face_img = frame[face_min[1]:face_max[1], face_min[0]:face_max[0]]
    
    # Prepare inputs based on model type
    if model_type == "standard":
        # Standard model only needs the face image
        face_input = preprocess_image(face_img, target_size)
        face_input = np.expand_dims(face_input, axis=0)  # Add batch dimension
        
        # Predict
        prediction = model.predict(face_input, verbose=0)[0]
        
    else:
        # For multi-input or head_pose models, we need more data
        
        # Extract eye regions
        left_eye_pts = landmarks[landmark_indices['left_eye']]
        right_eye_pts = landmarks[landmark_indices['right_eye']]
        
        # Calculate eye bounding boxes
        left_eye_min = np.min(left_eye_pts[:, :2], axis=0).astype(int)
        left_eye_max = np.max(left_eye_pts[:, :2], axis=0).astype(int)
        right_eye_min = np.min(right_eye_pts[:, :2], axis=0).astype(int)
        right_eye_max = np.max(right_eye_pts[:, :2], axis=0).astype(int)
        
        # Add margin
        eye_margin = 5
        left_eye_min = np.maximum(0, left_eye_min - eye_margin)
        left_eye_max = np.minimum([w, h], left_eye_max + eye_margin)
        right_eye_min = np.maximum(0, right_eye_min - eye_margin)
        right_eye_max = np.minimum([w, h], right_eye_max + eye_margin)
        
        # Extract eye images
        left_eye_img = frame[left_eye_min[1]:left_eye_max[1], left_eye_min[0]:left_eye_max[0]]
        right_eye_img = frame[right_eye_min[1]:right_eye_max[1], right_eye_min[0]:right_eye_max[0]]
        
        if model_type == "multi_input":
            # Preprocess all images
            face_input = preprocess_image(face_img, target_size)
            left_eye_input = preprocess_image(left_eye_img, target_size)
            right_eye_input = preprocess_image(right_eye_img, target_size)
            
            # Add batch dimension
            face_input = np.expand_dims(face_input, axis=0)
            left_eye_input = np.expand_dims(left_eye_input, axis=0)
            right_eye_input = np.expand_dims(right_eye_input, axis=0)
            
            # Prepare model inputs
            model_inputs = {
                'face_input': face_input,
                'left_eye_input': left_eye_input,
                'right_eye_input': right_eye_input
            }
            
            # Predict
            prediction = model.predict(model_inputs, verbose=0)[0]
            
        elif model_type == "head_pose":
            # Estimate head pose
            pitch, yaw, roll = estimate_head_pose(landmarks, (h, w))
            
            # Preprocess face image
            face_input = preprocess_image(face_img, target_size)
            face_input = np.expand_dims(face_input, axis=0)
            
            # Prepare head pose input
            head_pose_input = np.array([[pitch, yaw, roll]], dtype=np.float32)
            
            # Prepare model inputs
            model_inputs = {
                'face_input': face_input,
                'head_pose_input': head_pose_input
            }
            
            # Predict
            prediction = model.predict(model_inputs, verbose=0)[0]
    
    # Clip prediction to [0, 1] range
    prediction = np.clip(prediction, 0, 1)
    
    # Prepare result
    result = {
        "success": True,
        "image": orig_frame,
        "face": face_img,
        "prediction": prediction,
        "face_box": (face_min[0], face_min[1], face_max[0], face_max[1]),
        "landmarks": landmarks
    }
    
    # Add extra data based on model type
    if model_type == "multi_input":
        result["left_eye"] = left_eye_img
        result["right_eye"] = right_eye_img
        
    elif model_type == "head_pose":
        result["head_pose"] = (pitch, yaw, roll)
    
    return result

def visualize_realtime_result(
    result: Dict,
    frame: np.ndarray,
    fps: float = 0.0,
    gaze_history: List = None
) -> np.ndarray:
    """
    Visualize the processing result for real-time display.
    
    Args:
        result: Processing result from process_frame()
        frame: Original frame
        fps: Frames per second
        gaze_history: List of previous gaze points
        
    Returns:
        Annotated frame
    """
    if not result["success"]:
        # Display error on frame
        cv2.putText(
            frame, 
            f"Error: {result.get('error', 'Unknown error')}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        return frame
        
    # Create a copy of the frame
    display = frame.copy()
    
    # Draw face box
    x1, y1, x2, y2 = result["face_box"]
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw MediaPipe landmarks if available
    if "landmarks" in result:
        landmarks = result["landmarks"]
        for i, (x, y, _) in enumerate(landmarks):
            # Draw only a subset of landmarks to avoid cluttering
            if i % 10 == 0:
                cv2.circle(display, (int(x), int(y)), 1, (0, 255, 255), -1)
    
    # Convert normalized gaze coordinates to pixel coordinates
    h, w = display.shape[:2]
    pred_x, pred_y = result["prediction"]
    x_pixel = int(pred_x * w)
    y_pixel = int(pred_y * h)
    
    # Draw current gaze point
    cv2.circle(display, (x_pixel, y_pixel), 10, (0, 0, 255), -1)
    
    # Draw gaze history (trail)
    if gaze_history:
        for i, point in enumerate(gaze_history):
            alpha = (i + 1) / len(gaze_history)  # Opacity increases with newer points
            point_x = int(point[0] * w)
            point_y = int(point[1] * h)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            size = max(1, int(5 * alpha))
            cv2.circle(display, (point_x, point_y), size, color, -1)
    
    # Draw FPS
    cv2.putText(
        display, 
        f"FPS: {fps:.1f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 255, 0), 
        2
    )
    
    # Add head pose info if available
    if "head_pose" in result:
        pitch, yaw, roll = result["head_pose"]
        cv2.putText(
            display, 
            f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
    
    return display

def run_webcam_inference(
    model: tf.keras.Model,
    model_type: str,
    camera_id: int = 0,
    target_size: Tuple[int, int] = (64, 64),
    show_fps: bool = True,
    enable_recording: bool = False,
    record_path: str = "gaze_recording.mp4"
) -> None:
    """
    Run real-time gaze tracking on webcam feed.
    
    Args:
        model: Trained model
        model_type: Type of model
        camera_id: Camera ID to use
        target_size: Target size for image preprocessing
        show_fps: Whether to show FPS
        enable_recording: Whether to record the output
        record_path: Path to save the recording
    """
    # Initialize MediaPipe face mesh with streaming mode
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Define key landmark indices
    landmark_indices = {
        # Left eye landmarks
        'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        # Right eye landmarks
        'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        # Face landmarks for pose estimation
        'pose': [1, 33, 61, 199, 263, 291]  # Nose, eyes, mouth corners
    }
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened with resolution {width}x{height} at {fps} FPS")
    
    # Initialize video writer if recording is enabled
    writer = None
    if enable_recording:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(record_path, fourcc, fps, (width, height))
    
    # Create window
    cv2.namedWindow("Gaze Tracking", cv2.WINDOW_NORMAL)
    
    # Initialize variables for FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    
    # Initialize gaze history
    gaze_history = []
    max_history = 30
    
    # Create heatmap for visualization
    heatmap_size = (400, 300)
    heatmap = np.zeros((heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    heatmap_decay = 0.95
    
    cv2.namedWindow("Gaze Heatmap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Heatmap", heatmap_size[0], heatmap_size[1])
    
    print("Starting real-time gaze tracking. Press 'q' to quit, 's' to save a screenshot.")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Start timing
            start_time = time.time()
            
            # Process the frame
            result = process_frame(frame, model, model_type, face_mesh, landmark_indices, target_size)
            
            # Update gaze history if detection was successful
            if result["success"]:
                gaze_point = result["prediction"]
                gaze_history.append(gaze_point)
                if len(gaze_history) > max_history:
                    gaze_history.pop(0)
                
                # Update heatmap
                x_heatmap = int(gaze_point[0] * heatmap_size[0])
                y_heatmap = int(gaze_point[1] * heatmap_size[1])
                cv2.circle(heatmap, (x_heatmap, y_heatmap), 15, (5,), -1)
                
            # Apply decay to heatmap
            heatmap *= heatmap_decay
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 10:
                current_time = time.time()
                elapsed = current_time - fps_start_time
                current_fps = fps_counter / elapsed if elapsed > 0 else 0
                fps_counter = 0
                fps_start_time = current_time
            
            # Visualize the result
            display = visualize_realtime_result(result, frame, current_fps, gaze_history)
            
            # Display the frame
            cv2.imshow("Gaze Tracking", display)
            
            # Display heatmap
            heatmap_vis = cv2.applyColorMap(np.uint8(np.clip(heatmap * 10, 0, 255)), cv2.COLORMAP_JET)
            cv2.rectangle(heatmap_vis, (0, 0), (heatmap_size[0]-1, heatmap_size[1]-1), (0, 255, 0), 2)
            cv2.imshow("Gaze Heatmap", heatmap_vis)
            
            # Write frame if recording
            if writer is not None:
                writer.write(display)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"gaze_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, display)
                print(f"Screenshot saved to {filename}")
    
    finally:
        # Release resources
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        print("Gaze tracking stopped")

def process_directory(
    directory: str,
    model: tf.keras.Model,
    model_type: str,
    output_dir: str,
    target_size: Tuple[int, int] = (64, 64)
) -> None:
    """
    Process all images in a directory.
    
    Args:
        directory: Directory containing input images
        model: Trained model
        model_type: Type of model
        output_dir: Directory to save outputs
        target_size: Target size for image preprocessing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up MediaPipe
    face_mesh, landmark_indices = setup_mediapipe()
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images in {directory}")
    
    # Process each image
    results = []
    
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        result = process_image(img_path, model, model_type, face_mesh, landmark_indices, target_size)
        
        if result["success"]:
            # Save result visualization
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_result.png")
            
            visualize_result(result, output_path, show_landmarks=False)
            
            # Keep track of predictions
            results.append({
                "file": img_path,
                "prediction": result["prediction"].tolist()
            })
        else:
            print(f"Failed to process {img_path}: {result.get('error', 'Unknown error')}")
            
    # Close MediaPipe
    face_mesh.close()
    
    # Save all predictions to a file
    if results:
        import json
        predictions_path = os.path.join(output_dir, "predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"All predictions saved to {predictions_path}")

def visualize_result(
    result: Dict,
    output_path: Optional[str] = None,
    show_landmarks: bool = False
) -> None:
    """
    Visualize the processing result.
    
    Args:
        result: Processing result from process_image()
        output_path: Path to save the visualization
        show_landmarks: Whether to show face landmarks
    """
    if not result["success"]:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
        
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Original image with gaze point
    plt.subplot(1, 2, 1)
    img = result["image"]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    
    # Draw face box
    x1, y1, x2, y2 = result["face_box"]
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
    
    # Draw predicted gaze point
    h, w = img.shape[:2]
    pred_x, pred_y = result["prediction"]
    plt.plot(pred_x * w, pred_y * h, 'bo', markersize=10)
    
    plt.title(f"Predicted Gaze: ({pred_x:.3f}, {pred_y:.3f})")
    plt.axis('off')
    
    # Face image
    plt.subplot(1, 2, 2)
    face_rgb = cv2.cvtColor(result["face"], cv2.COLOR_BGR2RGB)
    plt.imshow(face_rgb)
    plt.title("Detected Face")
    plt.axis('off')
    
    # Add head pose if available
    if "head_pose" in result:
        pitch, yaw, roll = result["head_pose"]
        plt.figtext(0.5, 0.02, f"Head Pose: Pitch={pitch:.1f}°, Yaw={yaw:.1f}°, Roll={roll:.1f}°", 
                   ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved to {output_path}")
        
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Gaze tracking inference")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single input image")
    group.add_argument("--directory", type=str, help="Path to a directory of images")
    group.add_argument("--webcam", action="store_true", help="Use webcam for real-time inference")
    
    # Model options
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", type=str, 
                      choices=["standard", "multi_input", "head_pose"],
                      help="Type of model (detected automatically if not specified)")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output path or directory")
    
    # Webcam options
    parser.add_argument("--camera", type=int, default=0, help="Camera ID for webcam mode")
    parser.add_argument("--record", action="store_true", help="Record the webcam output")
    parser.add_argument("--record_path", type=str, default="gaze_recording.mp4", 
                      help="Path for recorded video")
    
    args = parser.parse_args()
    
    # Load the model
    model, detected_model_type = load_model(args.model)
    
    # Use user-specified model type or fall back to detected type
    model_type = args.model_type if args.model_type else detected_model_type
    
    if args.webcam:
        # Run real-time inference on webcam
        run_webcam_inference(
            model=model,
            model_type=model_type,
            camera_id=args.camera,
            enable_recording=args.record,
            record_path=args.record_path
        )
    
    elif args.image:
        # Set up MediaPipe for static image
        face_mesh, landmark_indices = setup_mediapipe()
        
        # Process a single image
        result = process_image(
            args.image, 
            model, 
            model_type, 
            face_mesh,
            landmark_indices
        )
        
        output_path = args.output if args.output else None
        visualize_result(result, output_path)
        
        # Close MediaPipe
        face_mesh.close()
        
    else:  # args.directory
        # Process a directory of images
        output_dir = args.output if args.output else "gaze_results"
        process_directory(
            args.directory,
            model,
            model_type,
            output_dir
        )

if __name__ == "__main__":
    main()
