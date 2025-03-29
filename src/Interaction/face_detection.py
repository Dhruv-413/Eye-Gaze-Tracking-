# face_detection.py
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector
from core.utils import draw_landmarks

class FaceDetectionConfig:
    """Configuration class for face detection visualization"""
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
    Enhanced face detection system with visualization capabilities
    
    Features:
    - Configurable visualization parameters
    - Connection drawing for face mesh
    - Resource management
    - Frame processing statistics
    """
    
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        config: Optional[FaceDetectionConfig] = None
    ):
        """
        Args:
            face_detector: Initialized FaceMeshDetector instance
            config: Visualization configuration
        """
        self.detector = face_detector
        self.config = config or FaceDetectionConfig()
        self._processing_stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'avg_processing_time': 0.0
        }

    def process_frame(
        self,
        frame: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], Optional[NDArray[np.float32]]]:
        """
        Process frame and return annotated frame with detection results
        
        Returns:
            Tuple containing:
            - Annotated frame
            - Landmark points (if detected)
        """
        self._processing_stats['total_frames'] += 1
        start_time = cv2.getTickCount()
        
        landmarks = self.detector.process_frame(frame)
        annotated_frame = frame.copy()
        
        if landmarks is not None:
            self._processing_stats['detected_frames'] += 1
            self._draw_landmarks(annotated_frame, landmarks)
            
        # Update performance metrics
        processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        self._update_stats(processing_time)
        
        return annotated_frame, landmarks

    def _draw_landmarks(
        self,
        frame: NDArray[np.uint8],
        landmarks: NDArray[np.float32]
    ) -> None:
        """Apply configured visualization to detected landmarks"""
        if self.config.draw_all_landmarks:
            draw_landmarks(
                frame,
                landmarks,
                color=self.config.landmark_color,
                radius=self.config.landmark_radius
            )
            
        if self.config.draw_connections:
            self._draw_face_connections(frame, landmarks)

    def _draw_face_connections(
        self,
        frame: NDArray[np.uint8],
        landmarks: NDArray[np.float32]
    ) -> None:
        """Draw face mesh connections using MediaPipe's topology"""
        # Simplified connection drawing - implement full mesh as needed
        connections = self.detector.mp_face_mesh.FACEMESH_TESSELATION
        for connection in connections:
            start_idx, end_idx = connection
            x1, y1 = landmarks[start_idx]
            x2, y2 = landmarks[end_idx]
            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                self.config.connection_color,
                self.config.connection_thickness
            )

    def _update_stats(self, processing_time: float) -> None:
        """Update performance statistics with exponential moving average"""
        alpha = 0.1  # Smoothing factor
        self._processing_stats['avg_processing_time'] = \
            (1 - alpha) * self._processing_stats['avg_processing_time'] + \
            alpha * processing_time

    @property
    def detection_rate(self) -> float:
        """Current face detection success rate"""
        if self._processing_stats['total_frames'] == 0:
            return 0.0
        return self._processing_stats['detected_frames'] / self._processing_stats['total_frames']

    @property
    def stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self._processing_stats,
            'detection_rate': self.detection_rate,
            'fps': 1.0 / self._processing_stats['avg_processing_time'] 
                   if self._processing_stats['avg_processing_time'] > 0 else 0
        }

def main() -> None:
    """Example usage with resource management and visualization"""
    config = FaceDetectionConfig(
        landmark_color=(255, 0, 0),
        connection_color=(0, 255, 0),
        draw_all_landmarks=True,
        draw_connections=True,
        landmark_radius=2,
        connection_thickness=1
    )

    with FaceMeshDetector() as detector:
        face_detection = FaceDetection(detector, config=config)
        
        with cv2.VideoCapture(0) as cap:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, landmarks = face_detection.process_frame(frame)
                
                # Display stats
                stats = face_detection.stats
                cv2.putText(annotated_frame, f"FPS: {stats['fps']:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Detection Rate: {stats['detection_rate']:.1%}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Face Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    main()