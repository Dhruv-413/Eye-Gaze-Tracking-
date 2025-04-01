# face_tracker.py
import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Optional, Tuple
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FaceMeshDetector:
    """
    Detects and processes facial landmarks for a single face using MediaPipe's FaceMesh.
    Provides methods to extract landmarks, draw them and crop eye regions.
    """
    def __init__(
        self,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self._validate_confidence(min_detection_confidence, "min_detection_confidence")
        self._validate_confidence(min_tracking_confidence, "min_tracking_confidence")
        self.mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False
        )
        logger.info("FaceMeshDetector initialized.")

    @staticmethod
    def _validate_confidence(value: float, name: str) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be between 0.0 and 1.0; got {value}.")

    @staticmethod
    def _is_valid_frame(frame: Optional[NDArray[np.uint8]]) -> bool:
        return frame is not None and frame.size > 0 and len(frame.shape) == 3 and frame.shape[2] == 3

    def process_frame(self, frame: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        if not self._is_valid_frame(frame):
            logger.warning("Received an invalid or empty frame.")
            return None
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                logger.debug("No facial landmarks detected in the frame.")
                return None
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            landmarks_pixels = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks],
                dtype=np.float32
            )
            logger.debug("Extracted landmarks successfully.")
            return landmarks_pixels
        except Exception as e:
            logger.error("Error processing frame: %s", e, exc_info=True)
            return None

    def draw_landmarks(
        self,
        frame: NDArray[np.uint8],
        landmarks: NDArray[np.float32],
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 2
    ) -> None:
        if landmarks is None or landmarks.size == 0:
            logger.debug("No landmarks provided for drawing.")
            return
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        logger.debug("Landmarks drawn on frame.")

    def draw_face_bounding_box(
        self,
        frame: NDArray[np.uint8],
        landmarks: NDArray[np.float32],
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> None:
        if landmarks is None or landmarks.size == 0:
            logger.debug("No landmarks available to draw a bounding box.")
            return
        x, y, w, h = cv2.boundingRect(landmarks.astype(np.int32))
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    def crop_eye_region(
        self,
        frame: NDArray[np.uint8],
        landmarks: NDArray[np.float32],
        eye_indices: NDArray[np.int32],
        padding_factor: float = 0.2
    ) -> Optional[NDArray[np.uint8]]:
        if not self._is_valid_frame(frame) or landmarks is None or landmarks.size == 0:
            logger.warning("Invalid frame or landmarks provided for eye region cropping.")
            return None
        try:
            eye_points = landmarks[eye_indices]
            x, y, w, h = cv2.boundingRect(eye_points.astype(np.int32))
            pad = int(padding_factor * max(w, h))
            x1 = max(0, x-pad)
            y1 = max(0, y-pad)
            x2 = min(frame.shape[1], x+w+pad)
            y2 = min(frame.shape[0], y+h+pad)
            if x1 >= x2 or y1 >= y2:
                logger.debug("Calculated eye region is invalid after padding.")
                return None
            return frame[y1:y2, x1:x2]
        except Exception as e:
            logger.error("Error cropping eye region: %s", e, exc_info=True)
            return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._face_mesh.close()
        if exc_type is not None:
            logger.error("Exception in context: %s", exc_value, exc_info=True)
