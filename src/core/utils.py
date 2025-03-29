# utils.py
import cv2
import numpy as np
from typing import Tuple, Sequence, Optional
from numpy.typing import NDArray

def calculate_ear(eye: NDArray[np.float32]) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for blink detection using 6 facial landmarks.
    
    Args:
        eye: Array of shape (6, 2) containing (x, y) coordinates of eye landmarks
        
    Returns:
        EAR value as float
        
    Raises:
        ValueError: If input array has incorrect shape
        
    Example:
        >>> eye_points = np.array([[x1,y1], [x2,y2], ..., [x6,y6]])
        >>> ear = calculate_ear(eye_points)
    """
    if eye.shape != (6, 2):
        raise ValueError(f"Expected eye shape (6, 2), got {eye.shape}")
        
    # Use efficient vector operations
    vertical1 = np.linalg.norm(eye[1] - eye[5])
    vertical2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    
    # Add epsilon to prevent division by zero
    return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)

def draw_landmarks(
    frame: NDArray[np.uint8],
    landmarks: NDArray[np.float32],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 1
) -> None:
    """
    Draw facial landmarks on a frame with optional styling
    
    Args:
        frame: Input image in BGR format
        landmarks: Array of shape (N, 2) containing (x, y) coordinates
        color: BGR color tuple
        radius: Radius of drawn circles
    """
    # Vectorized drawing operation
    cv2.polylines(frame, [landmarks.astype(np.int32)], 
                 isClosed=False, color=color, thickness=1)
    
    # Efficient batch drawing using list comprehension
    [cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA) 
     for (x, y) in landmarks]