import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
from numpy.typing import NDArray

class FaceMeshDetector:
    """
    Single-face mesh detector with essential tracking features
    
    Features:
    - Single face detection optimized for gaze tracking
    - Automatic resource management
    - Configurable performance parameters
    - Basic error handling
    """
    
    def __init__(
        self,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self._validate_config(min_detection_confidence)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Enforce single face
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False
        )

    def _validate_config(self, confidence: float):
        """Validate initialization parameters"""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0-1.0")

    def process_frame(self, frame: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        """
        Process a video frame and return facial landmarks
        
        Args:
            frame: Input BGR image (expected shape: [H, W, 3])
            
        Returns:
            Optional NDArray: Array of shape (468, 2) containing (x, y) coordinates
                              Returns None if no face detected
        """
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            print("FaceMeshDetector: Invalid frame input.")
            return None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                print("FaceMeshDetector: No face landmarks detected.")
                return None
                
            h, w = frame.shape[:2]
            landmarks = results.multi_face_landmarks[0].landmark
            
            return np.array([(lm.x * w, lm.y * h) for lm in landmarks], 
                           dtype=np.float32)
            
        except Exception as e:
            print(f"FaceMeshDetector: Error during face detection - {str(e)}")
            return None

    def draw_landmarks(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> None:
        """
        Draw facial landmarks on the frame.
        
        Args:
            frame: Input BGR image.
            landmarks: Array of shape (468, 2) containing (x, y) coordinates.
        """
        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, cv2.LINE_AA)

    def draw_face_bounding_box(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> None:
        """
        Draw a bounding box around the face.
        """
        x, y, w, h = cv2.boundingRect(landmarks.astype(np.int32))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def crop_eye_region(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32], eye_indices: NDArray[np.int32]) -> Optional[NDArray[np.uint8]]:
        """
        Crop the eye region based on landmarks.
        
        Args:
            frame: Input BGR image.
            landmarks: Array of shape (468, 2) containing (x, y) coordinates.
            eye_indices: Indices of the eye landmarks.
        
        Returns:
            Cropped eye region as an image or None if invalid.
        """
        try:
            eye_points = landmarks[eye_indices]
            x, y, w, h = cv2.boundingRect(eye_points.astype(np.int32))
            padding = int(0.2 * max(w, h))  # Add padding
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
            return frame[y1:y2, x1:x2]
        except Exception as e:
            print(f"Error cropping eye region: {e}")
            return None

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._face_mesh.close()