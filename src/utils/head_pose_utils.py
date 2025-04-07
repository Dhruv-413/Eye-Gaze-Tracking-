"""
Utilities for extracting and processing head pose data
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any

def estimate_head_pose_from_landmarks(landmarks: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate head pose (Euler angles) from 3D facial landmarks.
    
    Args:
        landmarks: 3D landmarks from MediaPipe Face Mesh
        
    Returns:
        Tuple of (pitch, yaw, roll) angles in degrees
    """
    # 3D model points (standard face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float32)
    
    # Camera matrix (approximated)
    focal_length = 500
    center = (256, 256)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float32
    )
    
    # Distortion coefficients
    dist_coeffs = np.zeros((4,1))
    
    # Map MediaPipe landmarks to 3D model points
    # Standard indices for face landmarks from MediaPipe
    landmark_indices = [1, 199, 33, 263, 61, 291]  # Nose tip, chin, left eye, right eye, left mouth, right mouth
    
    # Extract landmarks (if we have enough)
    if landmarks.shape[0] < max(landmark_indices) + 1:
        return 0.0, 0.0, 0.0  # Not enough landmarks
    
    image_points = np.array([
        landmarks[i][:2] for i in landmark_indices
    ], dtype=np.float32)
    
    # Convert z values to match model
    z_scale = 300  # Scale factor for Z
    model_points[:, 2] = [landmarks[i][2] * z_scale for i in landmark_indices]
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        return 0.0, 0.0, 0.0
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Convert rotation matrix to Euler angles (pitch, yaw, roll)
    # https://learnopencv.com/rotation-matrix-to-euler-angles/
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    
    # Convert radians to degrees
    pitch = np.rad2deg(x)
    yaw = np.rad2deg(y)
    roll = np.rad2deg(z)
    
    return pitch, yaw, roll

def preprocess_head_pose(pose_data: np.ndarray) -> np.ndarray:
    """
    Preprocess head pose data for model input.
    
    Args:
        pose_data: Raw head pose data as (N, 3) array (pitch, yaw, roll)
        
    Returns:
        Normalized head pose data
    """
    # Create a copy to avoid modifying the original
    processed = pose_data.copy()
    
    # Simple normalization to reasonable ranges
    # Typical ranges: pitch ±90°, yaw ±90°, roll ±45°
    processed[:, 0] /= 90.0  # Normalize pitch to roughly [-1, 1]
    processed[:, 1] /= 90.0  # Normalize yaw to roughly [-1, 1]
    processed[:, 2] /= 45.0  # Normalize roll to roughly [-1, 1]
    
    # Clip extreme values
    return np.clip(processed, -1.5, 1.5)

def extract_head_pose_from_face(face_image: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract head pose directly from a face image using MediaPipe.
    
    Args:
        face_image: Face image as numpy array
        
    Returns:
        Tuple of (pitch, yaw, roll) angles in degrees
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        # Convert to RGB
        image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return 0.0, 0.0, 0.0  # No face detected
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        h, w = face_image.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]  # Scale z by width for reasonable values
            for lm in face_landmarks.landmark
        ])
        
        # Get head pose
        return estimate_head_pose_from_landmarks(landmarks)
