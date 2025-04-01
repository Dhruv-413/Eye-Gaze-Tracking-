# feedback.py
import cv2
import numpy as np
import time
import logging
from collections import deque
from typing import NamedTuple, Optional, Dict, Any, Tuple
from numpy.typing import NDArray

from core.face_tracker import FaceMeshDetector
from Interaction.eye_blink import EyeBlinkDetector
from Interaction.constants import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeedbackResult(NamedTuple):
    frame: NDArray[np.uint8]
    landmarks: Optional[NDArray[np.float32]]
    ear: float
    blinks: int
    system_status: Dict[str, Any]
    timestamp: float

class FeedbackConfig:
    def __init__(
        self,
        text_color: Tuple[int, int, int] = (0, 255, 0),
        warning_color: Tuple[int, int, int] = (0, 0, 255),
        font_scale: float = 0.7,
        font_thickness: int = 2,
        text_margin: int = 20,
        show_metrics: bool = True,
        show_landmarks: bool = True
    ):
        self.text_color = text_color
        self.warning_color = warning_color
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_margin = text_margin
        self.show_metrics = show_metrics
        self.show_landmarks = show_landmarks

class FeedbackSystem:
    """
    Comprehensive feedback system with real-time metrics and system status overlay.
    """
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        blink_detector: EyeBlinkDetector,
        config: Optional[FeedbackConfig] = None
    ):
        self.detector = face_detector
        self.blink_detector = blink_detector
        self.config = config or FeedbackConfig()
        self._metrics = {'frame_count': 0, 'processing_times': deque(maxlen=100), 'system_status': 'INITIALIZING'}
        logger.info("FeedbackSystem initialized.")

    def process_frame(self, frame: NDArray[np.uint8]) -> FeedbackResult:
        self._metrics['frame_count'] += 1
        start_time = time.time()
        landmarks = self.detector.process_frame(frame)
        blink_result = self.blink_detector.process_frame(frame)
        processing_time = time.time() - start_time
        self._metrics['processing_times'].append(processing_time)
        system_status = self._get_system_status()
        return FeedbackResult(
            frame=frame.copy(),
            landmarks=landmarks,
            ear=blink_result.get('ear', 0.0),
            blinks=blink_result.get('blink_count', 0),
            system_status=system_status,
            timestamp=time.time()
        )

    def draw_analytics(self, result: FeedbackResult) -> NDArray[np.uint8]:
        annotated_frame = result.frame.copy()
        y_offset = self.config.text_margin
        self._draw_text(annotated_frame, f"EAR: {result.ear:.2f}", y_offset)
        self._draw_text(annotated_frame, f"Blinks: {result.blinks}", y_offset + 30)
        if self.config.show_metrics:
            self._draw_metrics(annotated_frame, result)
        if self.config.show_landmarks and result.landmarks is not None:
            self._draw_landmarks(annotated_frame, result.landmarks)
        return annotated_frame

    def _draw_landmarks(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> None:
        try:
            for idx in [*LEFT_EYE_EAR_IDX, *RIGHT_EYE_EAR_IDX]:
                x, y = landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), 2, self.config.text_color, -1)
        except Exception as e:
            logger.error("Error drawing landmarks: %s", e, exc_info=True)

    def _draw_metrics(self, frame: NDArray[np.uint8], result: FeedbackResult) -> None:
        y_start = self.config.text_margin + 60
        metrics = [
            f"FPS: {1/self._get_avg_processing_time():.1f}",
            f"Detection Rate: {result.system_status.get('detection_rate', 0):.1%}"
        ]
        line_spacing = 30
        for i, metric in enumerate(metrics):
            self._draw_text(frame, metric, y_start + i * line_spacing)
        self._draw_text(frame, f"Status: {result.system_status.get('status', 'UNKNOWN')}",
                        y_start + len(metrics) * line_spacing, self.config.warning_color)

    def _draw_text(self, frame: NDArray[np.uint8], text: str, y_pos: int, color: Optional[Tuple[int, int, int]] = None) -> None:
        cv2.putText(frame, text, (self.config.text_margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale, color or self.config.text_color, self.config.font_thickness, cv2.LINE_AA)

    def _get_system_status(self) -> Dict[str, Any]:
        avg_time = self._get_avg_processing_time()
        fps = 1 / avg_time if avg_time > 0 else 0.0
        detection_rate = self.blink_detector.detection_rate
        current_threshold = self.blink_detector.ear_threshold or getattr(self.blink_detector, "_dynamic_threshold", 0.0)
        return {
            'status': self._determine_system_state(),
            'fps': fps,
            'detection_rate': detection_rate,
            'ear_threshold': current_threshold,
            'frame_latency': avg_time
        }

    def _determine_system_state(self) -> str:
        detection_rate = self.blink_detector.detection_rate
        avg_latency = self._get_avg_processing_time()
        if detection_rate < 0.5:
            return "LOW CONFIDENCE (Check Lighting)"
        if avg_latency > 0.1:
            return "HIGH LATENCY"
        return "NORMAL OPERATION"

    def _get_avg_processing_time(self) -> float:
        return np.mean(self._metrics['processing_times']) if self._metrics['processing_times'] else 0.0

    def get_feedback(self, blink_result: dict, gaze_result: Any, pose_result: Any) -> str:
        feedback_lines = []
        if blink_result.get("blink_detected", False):
            feedback_lines.append("Blink detected.")
        else:
            feedback_lines.append("No blink detected.")
        if pose_result and getattr(pose_result, "confidence", 0) > 0.3:
            feedback_lines.append("Head pose:")
            feedback_lines.append(f"  Pitch: {pose_result.euler_angles[0]:.1f}")
            feedback_lines.append(f"  Yaw: {pose_result.euler_angles[1]:.1f}")
            feedback_lines.append(f"  Roll: {pose_result.euler_angles[2]:.1f}")
        else:
            feedback_lines.append("Head pose not detected or low confidence.")
        return "\n".join(feedback_lines)
