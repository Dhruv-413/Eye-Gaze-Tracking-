# feedback.py
import cv2
import numpy as np
from collections import deque
from typing import NamedTuple, Optional, Dict, Any, Tuple
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector
from core.utils import calculate_ear
from Interaction.eye_blink import EyeBlinkDetector  # Reuse existing component
from Interaction.constants import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX

class FeedbackResult(NamedTuple):
    frame: NDArray[np.uint8]
    landmarks: Optional[NDArray[np.float32]]
    ear: float
    blinks: int
    system_status: Dict[str, Any]
    timestamp: float

class FeedbackConfig:
    """Configuration for visual feedback parameters"""
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
    Comprehensive feedback system with multiple monitoring capabilities
    
    Features:
    - Real-time performance metrics
    - Configurable visualization
    - System health monitoring
    - Temporal data smoothing
    - Structured output format
    """
    
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        blink_detector: EyeBlinkDetector,
        config: Optional[FeedbackConfig] = None
    ):
        """
        Args:
            face_detector: Initialized FaceMeshDetector
            blink_detector: Configured EyeBlinkDetector
            config: Feedback visualization configuration
        """
        self.detector = face_detector
        self.blink_detector = blink_detector
        self.config = config or FeedbackConfig()
        
        # Performance tracking
        self._metrics = {
            'frame_count': 0,
            'processing_times': deque(maxlen=100),
            'detection_rates': deque(maxlen=100),
            'system_status': 'INITIALIZING'
        }

    def process_frame(self, frame: NDArray[np.uint8]) -> FeedbackResult:
        """Process frame and generate feedback data"""
        start_time = cv2.getTickCount()
        self._metrics['frame_count'] += 1
        
        # Process frame through pipeline
        landmarks = self.detector.process_frame(frame)
        blink_result = self.blink_detector.process_frame(frame)
        
        # Calculate metrics
        processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        self._metrics['processing_times'].append(processing_time)
        
        # Prepare result structure
        return FeedbackResult(
            frame=frame.copy(),
            landmarks=landmarks,
            ear=blink_result.get('ear', 0.0),
            blinks=blink_result.get('blink_count', 0),
            system_status=self._get_system_status(),
            timestamp=cv2.getTickCount() / cv2.getTickFrequency()
        )

    def draw_analytics(self, result: FeedbackResult) -> NDArray[np.uint8]:
        """Apply visual feedback to frame."""
        annotated_frame = result.frame.copy()
        y_offset = 30
        cv2.putText(annotated_frame, f"EAR: {result.ear:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Blinks: {result.blinks}", (10, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated_frame

    def _draw_landmarks(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> None:
        """Draw facial landmarks with configurable style"""
        for idx in [*LEFT_EYE_EAR_IDX, *RIGHT_EYE_EAR_IDX]:
            x, y = landmarks[idx]
            cv2.circle(frame, (int(x), int(y)), 2, self.config.text_color, -1)

    def _draw_metrics(self, frame: NDArray[np.uint8], result: FeedbackResult) -> None:
        """Draw system metrics and status information"""
        y_start = self.config.text_margin
        line_spacing = 30
        
        # System status
        self._draw_text(
            frame, f"Status: {result.system_status['status']}", 
            y_start, self.config.warning_color
        )
        
        # Performance metrics
        metrics = [
            f"FPS: {1/self._get_avg_processing_time():.1f}",
            f"EAR: {result.ear:.2f} (Thresh: {result.system_status['ear_threshold']:.2f})",
            f"Blinks: {result.blinks}",
            f"Detection Rate: {result.system_status['detection_rate']:.1%}"
        ]
        
        for i, metric in enumerate(metrics):
            self._draw_text(frame, metric, y_start + (i+1)*line_spacing)

    def _draw_text(self, frame: NDArray[np.uint8], text: str, y_pos: int, 
                 color: Optional[Tuple[int, int, int]] = None) -> None:
        """Helper method for consistent text drawing"""
        cv2.putText(
            frame, text,
            (self.config.text_margin, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            color or self.config.text_color,
            self.config.font_thickness,
            cv2.LINE_AA
        )

    def _get_system_status(self) -> Dict[str, Any]:
        """Calculate current system health metrics"""
        return {
            'status': self._determine_system_state(),
            'fps': 1 / self._get_avg_processing_time(),
            'detection_rate': self.blink_detector.detection_rate,
            'ear_threshold': self.blink_detector.current_threshold,
            'frame_latency': self._get_avg_processing_time()
        }

    def _determine_system_state(self) -> str:
        """Determine human-readable system state"""
        detection_rate = self.blink_detector.detection_rate
        if detection_rate < 0.5:
            return "LOW CONFIDENCE (Check Lighting)"
        if self._get_avg_processing_time() > 0.1:
            return "HIGH LATENCY"
        return "NORMAL OPERATION"

    def _get_avg_processing_time(self) -> float:
        """Calculate smoothed average processing time"""
        if not self._metrics['processing_times']:
            return 0.0
        return np.mean(self._metrics['processing_times'])

    def get_feedback(self, blink_result: dict, gaze_result: Any, pose_result: Any) -> str:
        """
        Generate feedback text based on blink, gaze, and head pose results.

        Args:
            blink_result: Blink detection result dictionary.
            gaze_result: Gaze tracking result object.
            pose_result: Head pose estimation result object.

        Returns:
            Feedback text as a string.
        """
        feedback = []

        # Blink feedback
        if blink_result.get("blink_detected", False):
            feedback.append("Blink detected.")
        else:
            feedback.append("No blink detected.")

        # Head pose feedback
        if pose_result and pose_result.confidence > 0.3:  # Lowered confidence threshold
            feedback.append(f"Head pose:")
            feedback.append(f"  Pitch: {pose_result.euler_angles[0]:.1f}")
            feedback.append(f"  Yaw: {pose_result.euler_angles[1]:.1f}")
            feedback.append(f"  Roll: {pose_result.euler_angles[2]:.1f}")
        else:
            feedback.append("Head pose not detected or low confidence.")

        return "\n".join(feedback)

def main() -> None:
    """Example usage with resource management"""
    config = FeedbackConfig(
        text_color=(0, 255, 255),
        warning_color=(0, 0, 255),
        font_scale=0.8,
        show_metrics=True,
        show_landmarks=True
    )

    with FaceMeshDetector() as face_detector:
        blink_detector = EyeBlinkDetector(face_detector)
        feedback_system = FeedbackSystem(face_detector, blink_detector, config)
        
        with cv2.VideoCapture(0) as cap:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame through pipeline
                result = feedback_system.process_frame(frame)
                annotated_frame = feedback_system.draw_analytics(result)
                
                # Display results
                cv2.imshow("Feedback System", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    main()