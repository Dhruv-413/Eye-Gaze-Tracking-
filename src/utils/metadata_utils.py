import numpy as np
import math
import time
from typing import Dict, Tuple, List, Optional, Any
import cv2

def extract_eye_coords(eye_data: Any) -> Optional[Dict[str, float]]:
    """Extract eye coordinates from MediaPipe eye_data"""
    if not eye_data or not hasattr(eye_data, 'landmarks') or not eye_data.landmarks:
        return None
        
    # Extract bounding box from landmarks
    x_coords = [p[0] for p in eye_data.landmarks]
    y_coords = [p[1] for p in eye_data.landmarks]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    return {
        "X": min_x,
        "Y": min_y,
        "W": max_x - min_x,
        "H": max_y - min_y
    }

def generate_metadata(
    frame_shape: Tuple[int, int, int],
    face_landmarks: Any,
    left_eye_data: Any,
    right_eye_data: Any,
    screen_width: int,
    screen_height: int,
    frame_time: float,
    head_pose: Optional[Dict[str, Any]] = None  # Add head_pose parameter
) -> Dict[str, Any]:
    """
    Generate metadata similar to the training dataset format
    
    Args:
        frame_shape: Shape of the input frame (h, w, c)
        face_landmarks: MediaPipe face landmarks
        left_eye_data: MediaPipe left eye data
        right_eye_data: MediaPipe right eye data
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        frame_time: Timestamp for the current frame
        head_pose: Optional head pose information from solvePnP
        
    Returns:
        Dictionary containing metadata similar to training format
    """
    frame_height, frame_width = frame_shape[:2]
    
    # Extract left and right eye coordinates
    left_eye_coords = extract_eye_coords(left_eye_data)
    right_eye_coords = extract_eye_coords(right_eye_data)
    
    # Calculate eyes midpoint
    eyes_midpoint = None
    if (left_eye_data and right_eye_data and 
        hasattr(left_eye_data, 'center') and hasattr(right_eye_data, 'center') and
        left_eye_data.center and right_eye_data.center):
        mid_x = (left_eye_data.center[0] + right_eye_data.center[0]) / 2
        mid_y = (left_eye_data.center[1] + right_eye_data.center[1]) / 2
        eyes_midpoint = {
            "X": mid_x,
            "Y": mid_y,
            "normalized_X": mid_x / frame_width,
            "normalized_Y": mid_y / frame_height
        }
    
    # Calculate face grid data (25x25 grid representation)
    face_grid_data = {"IsValid": 0}
    if face_landmarks is not None and hasattr(face_landmarks, 'landmark'):
        try:
            # Extract face bounding box
            face_landmarks_array = np.array([[landmark.x * frame_width, 
                                            landmark.y * frame_height] 
                                            for landmark in face_landmarks.landmark])
            x_min, y_min = np.min(face_landmarks_array, axis=0)
            x_max, y_max = np.max(face_landmarks_array, axis=0)
            face_width = x_max - x_min
            face_height = y_max - y_min
            
            # Convert to 25x25 grid coordinates
            grid_x = int((x_min / frame_width) * 25)
            grid_y = int((y_min / frame_height) * 25)
            grid_w = max(1, int((face_width / frame_width) * 25))
            grid_h = max(1, int((face_height / frame_height) * 25))
            
            face_grid_data = {
                "X": grid_x,
                "Y": grid_y,
                "W": grid_w,
                "H": grid_h,
                "IsValid": 1,
                "normalized_X": grid_x / 25.0,
                "normalized_Y": grid_y / 25.0,
                "normalized_W": grid_w / 25.0,
                "normalized_H": grid_h / 25.0
            }
        except Exception as e:
            print(f"Error calculating face grid: {e}")
    
    # Use provided head pose if available, otherwise calculate from eye positions
    if head_pose is not None:
        head_pose_data = {
            "pitch": head_pose.get("pitch", 0.0),
            "yaw": head_pose.get("yaw", 0.0),
            "roll": head_pose.get("roll", 0.0),
            "eye_distance": head_pose.get("eye_distance", 0.0) 
                if "eye_distance" in head_pose else
                calculate_eye_distance(left_eye_coords, right_eye_coords),
            "estimated_from_face": True
        }
    else:
        # Calculate head pose from eye positions as fallback
        head_pose_data = calculate_head_pose_from_eyes(
            left_eye_coords, right_eye_coords, eyes_midpoint, frame_width, frame_height
        )
    
    # Create metadata structure
    metadata = {
        "left_eye_coords": left_eye_coords,
        "right_eye_coords": right_eye_coords,
        "eyes_midpoint": eyes_midpoint,
        "dot_data": {
            "DotNum": 0,  # Placeholder
            "Time": frame_time,
            "normalized_X": 0.5,  # Will be updated during calibration/tracking
            "normalized_Y": 0.5   # Will be updated during calibration/tracking
        },
        "face_grid_data": face_grid_data,
        "motion_data": {
            "Time": frame_time,
            "DotNum": 0
        },
        "head_pose": head_pose_data,
        "screen_data": {
            "H": screen_height,
            "W": screen_width,
            "Orientation": 1  # Assuming landscape orientation for laptop
        },
        "image_dimensions": {
            "width": frame_width,
            "height": frame_height
        }
    }
    
    return metadata

def calculate_eye_distance(left_eye_coords, right_eye_coords):
    """Calculate distance between eyes"""
    if not left_eye_coords or not right_eye_coords:
        return 0.0
        
    # Calculate eye centers
    left_eye_center_x = left_eye_coords["X"] + left_eye_coords["W"] / 2
    left_eye_center_y = left_eye_coords["Y"] + left_eye_coords["H"] / 2
    right_eye_center_x = right_eye_coords["X"] + right_eye_coords["W"] / 2
    right_eye_center_y = right_eye_coords["Y"] + right_eye_coords["H"] / 2
    
    # Calculate eye distance
    return math.sqrt((right_eye_center_x - left_eye_center_x)**2 + 
                    (right_eye_center_y - left_eye_center_y)**2)

def calculate_head_pose_from_eyes(left_eye_coords, right_eye_coords, eyes_midpoint, frame_width, frame_height):
    """Calculate approximate head pose from eye coordinates"""
    if not left_eye_coords or not right_eye_coords:
        return {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "eye_distance": 0.0,
            "estimated_from_face": True
        }
    
    # Calculate eye centers
    left_eye_center_x = left_eye_coords["X"] + left_eye_coords["W"] / 2
    left_eye_center_y = left_eye_coords["Y"] + left_eye_coords["H"] / 2
    right_eye_center_x = right_eye_coords["X"] + right_eye_coords["W"] / 2
    right_eye_center_y = right_eye_coords["Y"] + right_eye_coords["H"] / 2
    
    # Calculate eye distance (interpupillary distance)
    eye_distance = math.sqrt((right_eye_center_x - left_eye_center_x)**2 + 
                            (right_eye_center_y - left_eye_center_y)**2)
    
    # Calculate the angle between eyes (roll)
    if right_eye_center_x != left_eye_center_x:
        roll = math.atan2(right_eye_center_y - left_eye_center_y, 
                        right_eye_center_x - left_eye_center_x)
    else:
        roll = 0
    
    # Estimate yaw from the horizontal position of eyes midpoint
    if eyes_midpoint:
        yaw_estimate = ((eyes_midpoint["normalized_X"] - 0.5) * math.pi/2)
        
        # Estimate pitch from vertical position of eyes midpoint
        pitch_estimate = ((eyes_midpoint["normalized_Y"] - 0.5) * math.pi/4)
        
        return {
            "pitch": pitch_estimate,
            "yaw": yaw_estimate,
            "roll": roll,
            "eye_distance": eye_distance,
            "estimated_from_face": True
        }
    else:
        return {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": roll,
            "eye_distance": eye_distance,
            "estimated_from_face": True
        }

def extract_metadata_features(metadata: Dict[str, Any]) -> np.ndarray:
    """
    Extract metadata features in the format expected by the model
    Similar to dataset.py's extract_metadata_features
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Numpy array of metadata features
    """
    features = []
    
    # Add screen info
    features.append(metadata["screen_data"]["W"] / 1000.0)  # Normalize width
    features.append(metadata["screen_data"]["H"] / 1000.0)  # Normalize height
    features.append(metadata["screen_data"]["Orientation"] / 4.0)  # Normalize orientation
    
    # Add face grid data
    features.append(metadata["face_grid_data"].get("normalized_X", 0.0))
    features.append(metadata["face_grid_data"].get("normalized_Y", 0.0))
    features.append(metadata["face_grid_data"].get("normalized_W", 0.0))
    features.append(metadata["face_grid_data"].get("normalized_H", 0.0))
    
    # Add time and DotNum
    features.append(metadata["motion_data"].get("Time", 0.0))
    features.append(metadata["motion_data"].get("DotNum", 0.0) / 23.0)  # Normalize
    
    # Add head pose data
    if "head_pose" in metadata and metadata["head_pose"]:
        features.append(metadata["head_pose"].get("pitch", 0.0))
        features.append(metadata["head_pose"].get("yaw", 0.0))
        features.append(metadata["head_pose"].get("roll", 0.0))
        features.append(metadata["head_pose"].get("eye_distance", 0.0) / 200.0)  # Normalize
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Add eyes midpoint data
    if "eyes_midpoint" in metadata and metadata["eyes_midpoint"]:
        features.append(metadata["eyes_midpoint"].get("normalized_X", 0.5))
        features.append(metadata["eyes_midpoint"].get("normalized_Y", 0.5))
    else:
        features.extend([0.5, 0.5])
    
    # Ensure we have exactly 15 features (padding if necessary)
    while len(features) < 15:
        features.append(0.0)
    
    return np.array(features[:15], dtype=np.float32)  # Take only the first 15 features
