import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any, NamedTuple
from numpy.typing import NDArray
from dataclasses import dataclass
from config import FaceLandmarks
from utils.logging_utils import configure_logging
from utils.image_utils import preprocess_image

logger = configure_logging("eye.log")

@dataclass
class EyeRegion:
    """Data class to store eye region information."""
    image: NDArray[np.uint8]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h in original frame
    landmarks: NDArray[np.float32]  # Eye landmarks in original frame
    relative_landmarks: NDArray[np.float32]  # Landmarks relative to the cropped image
    aspect_ratio: float
    is_closed: bool

class EyeProcessor:
    def __init__(self, closed_threshold: float = 0.2, min_eye_size: int = 10):
        self.closed_threshold = closed_threshold
        self.min_eye_size = min_eye_size
        self.left_eye_indices = FaceLandmarks.LEFT_EYE
        self.right_eye_indices = FaceLandmarks.RIGHT_EYE
    
    def __init__(self, 
                 min_eye_size: int = 10):
        """
        Initialize the eye processor.
        
        Args:
            closed_threshold: Threshold for eye aspect ratio to consider eye closed
            min_eye_size: Minimum size (width or height) for valid eye region
        """
        self._validate_min_size(min_eye_size)
        self.min_eye_size = min_eye_size

    @staticmethod
    def _validate_threshold(value: float) -> None:
        """Validate that threshold value is appropriate."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Threshold must be a number; got {type(value).__name__}.")
        if not (0.0 < value < 1.0):
            raise ValueError(f"Threshold should be between 0.0 and 1.0; got {value}.")
    
    @staticmethod
    def _validate_min_size(value: int) -> None:
        """Validate minimum eye size is positive."""
        if not isinstance(value, int):
            raise TypeError(f"Minimum eye size must be an integer; got {type(value).__name__}.")
        if value <= 0:
            raise ValueError(f"Minimum eye size must be positive; got {value}.")

    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:
        """Check if a frame is valid for processing."""
        return (isinstance(frame, np.ndarray) and 
                frame.size > 0 and 
                len(frame.shape) == 3)

    def crop_eye_region(self,
                         frame: NDArray[np.uint8],
                         landmarks: NDArray[np.float32],
                         eye_indices: NDArray[np.int32],
                         padding_factor: float = 0.2) -> Optional[EyeRegion]:
        """
        Crops the eye region from the frame based on the provided landmark indices.
        
        Args:
            frame: The input image (BGR)
            landmarks: Array of facial landmark coordinates
            eye_indices: Indices corresponding to the eye landmarks
            padding_factor: Fraction of the eye size to add as padding
            
        Returns:
            EyeRegion object containing the cropped image and metadata, or None if cropping fails
        """
        if not self._is_valid_frame(frame):
            logger.warning(f"Invalid frame provided (shape: {getattr(frame, 'shape', 'unknown')})")
            return None
            
        if landmarks is None or landmarks.size == 0:
            logger.warning("No landmarks provided for eye region cropping")
            return None
            
        try:
            # Select eye points from landmarks
            if not isinstance(eye_indices, np.ndarray):
                eye_indices = np.array(eye_indices)
                
            if eye_indices.size == 0 or np.max(eye_indices) >= landmarks.shape[0]:
                logger.warning(f"Invalid eye indices (max index: {np.max(eye_indices) if eye_indices.size > 0 else 'empty'}, landmarks size: {landmarks.shape[0]})")
                return None
                
            eye_points = landmarks[eye_indices]
            
            # Calculate bounding box with padding
            x, y, w, h = cv2.boundingRect(eye_points.astype(np.int32))
            
            # Validate eye size before padding
            if w < self.min_eye_size or h < self.min_eye_size:
                logger.debug(f"Eye region too small: {w}x{h}, minimum required: {self.min_eye_size}")
                return None
                
            # Apply padding
            pad_w = int(padding_factor * w)
            pad_h = int(padding_factor * h)
            
            # Ensure we maintain aspect ratio with padding
            max_pad = max(pad_w, pad_h)
            
            x1 = max(0, x - max_pad)
            y1 = max(0, y - max_pad)
            x2 = min(frame.shape[1], x + w + max_pad)
            y2 = min(frame.shape[0], y + h + max_pad)
            
            if x1 >= x2 or y1 >= y2:
                logger.debug(f"Calculated eye region is invalid after applying padding: ({x1}, {y1}) to ({x2}, {y2})")
                return None
                
            # Crop the eye region
            cropped = frame[y1:y2, x1:x2].copy()  # Create a copy to avoid reference issues
            
            # Calculate landmarks relative to the cropped image
            relative_landmarks = eye_points.copy()
            relative_landmarks[:, 0] = relative_landmarks[:, 0] - x1
            relative_landmarks[:, 1] = relative_landmarks[:, 1] - y1
            
            # Calculate eye aspect ratio
            ear = self.calculate_eye_aspect_ratio(eye_points)
            is_closed = ear < self.closed_threshold if ear is not None else False
            
            # Create result
            result = EyeRegion(
                image=cropped,
                bounding_box=(x1, y1, x2 - x1, y2 - y1),
                landmarks=eye_points,
                relative_landmarks=relative_landmarks,
                aspect_ratio=ear if ear is not None else 0.0,
                is_closed=is_closed
            )
            
            logger.debug(f"Eye region cropped successfully: {cropped.shape}, EAR: {ear:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error cropping eye region: {e}", exc_info=True)
            return None
        
    def draw_eye_landmarks(self, 
                         frame: NDArray[np.uint8],
                         eye_region: Optional[EyeRegion],
                         color: Tuple[int, int, int] = (0, 255, 0),
                         closed_color: Tuple[int, int, int] = (0, 0, 255),
                         thickness: int = 1,
                         radius: int = 1,
                         draw_aspect_ratio: bool = True) -> None:
        """
        Draw eye landmarks and information on frame.
        
        Args:
            frame: Image to draw on
            eye_region: EyeRegion object with landmarks and metadata
            color: Color for open eye (BGR)
            closed_color: Color for closed eye (BGR)
            thickness: Line thickness
            radius: Radius of landmark points
            draw_aspect_ratio: Whether to display the eye aspect ratio
        """
        if frame is None or eye_region is None:
            return
            
        # Choose color based on whether eye is closed
        current_color = closed_color if eye_region.is_closed else color
        
        # Get original frame coordinates
        x, y, w, h = eye_region.bounding_box
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), current_color, thickness)
        
        # Draw eye landmarks
        for point in eye_region.landmarks:
            cv2.circle(frame, (int(point[0]), int(point[1])), radius, current_color, -1)
        
        # Connect landmarks to form eye contour
        landmarks = eye_region.landmarks.astype(np.int32)
        cv2.polylines(frame, [landmarks], True, current_color, thickness, cv2.LINE_AA)
        
        # Draw aspect ratio

    def preprocess_eye_for_analysis(self, eye_region: Optional[EyeRegion], 
                                     target_size: Tuple[int, int] = (64, 32)) -> Optional[NDArray[np.uint8]]:
        """
        Preprocess eye image for further analysis (like gaze tracking or iris detection).
        
        Args:
            eye_region: EyeRegion object containing eye image and metadata
            target_size: Target size for the normalized eye image
            
        Returns:
            Preprocessed eye image or None if preprocessing fails
        """
        if eye_region is None or eye_region.image is None:
            return None
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(eye_region.image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to improve contrast
            equalized = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            
            # Resize to target size
            resized = preprocess_image(blurred, target_size)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error preprocessing eye image: {e}")
            return None