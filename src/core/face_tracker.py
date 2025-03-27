import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

class FaceMeshDetector:
    """
    Face mesh detector using MediaPipe.
    """
    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces, refine_landmarks=refine_landmarks
        )
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process the frame and return facial landmarks as an array.

        Args:
            frame (np.ndarray): BGR image.

        Returns:
            Optional[np.ndarray]: Array of (x, y) landmark points or None if no face is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        landmarks_points = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
        return landmarks_points
