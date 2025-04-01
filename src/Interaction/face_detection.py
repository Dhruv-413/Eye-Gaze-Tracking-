# face_detection.py
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector
from core.utils import draw_landmarks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FaceDetectionConfig:
    def __init__(
        self,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 0, 0),
        landmark_radius: int = 1,
        connection_thickness: int = 1,
        draw_all_landmarks: bool = True,
        draw_connections: bool = False
    ):
        self.landmark_color = landmark_color
        self.connection_color = connection_color
        self.landmark_radius = landmark_radius
        self.connection_thickness = connection_thickness
        self.draw_all_landmarks = draw_all_landmarks
        self.draw_connections = draw_connections

class FaceDetection:
    """
    Enhanced face detection system with configurable visualization and performance metrics.
    """
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        config: Optional[FaceDetectionConfig] = None
    ):
        self.detector = face_detector
        self.config = config or FaceDetectionConfig()
        self._processing_stats = {'total_frames': 0, 'detected_frames': 0, 'avg_processing_time': 0.0}
        logger.info("FaceDetection initialized.")

    def process_frame(self, frame: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], Optional[NDArray[np.float32]]]:
        self._processing_stats['total_frames'] += 1
        start_tick = cv2.getTickCount()
        landmarks = self.detector.process_frame(frame)
        annotated_frame = frame.copy()
        if landmarks is not None:
            self._processing_stats['detected_frames'] += 1
            self._draw_landmarks(annotated_frame, landmarks)
        processing_time = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
        self._update_stats(processing_time)
        return annotated_frame, landmarks

    def _draw_landmarks(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> None:
        if self.config.draw_all_landmarks:
            draw_landmarks(frame, landmarks, color=self.config.landmark_color, radius=self.config.landmark_radius)
        if self.config.draw_connections:
            self._draw_face_connections(frame, landmarks)

    def _draw_face_connections(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> None:
        connections = self.detector.mp_face_mesh.FACEMESH_TESSELATION
        for connection in connections:
            start_idx, end_idx = connection
            try:
                x1, y1 = landmarks[start_idx]
                x2, y2 = landmarks[end_idx]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.config.connection_color, self.config.connection_thickness)
            except Exception as e:
                logger.warning("Error drawing connection %s: %s", connection, e)

    def _update_stats(self, processing_time: float) -> None:
        alpha = 0.1
        self._processing_stats['avg_processing_time'] = ((1 - alpha) * self._processing_stats['avg_processing_time'] +
                                                         alpha * processing_time)

    @property
    def detection_rate(self) -> float:
        total = self._processing_stats['total_frames']
        return self._processing_stats['detected_frames'] / total if total else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        avg_time = self._processing_stats['avg_processing_time']
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        return {**self._processing_stats, 'detection_rate': self.detection_rate, 'fps': fps}
