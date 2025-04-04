# core/face.py
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Any
from numpy.typing import NDArray
from dataclasses import dataclass

# Import the default landmark configuration from constants
from config import DEFAULT, LandmarkIndices
from utils.logging_utils import configure_logging

logger = configure_logging("face.log")

@dataclass
class FaceDetectionResult:
    """Data class to store face detection results and metadata."""
    landmarks: NDArray[np.float32]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    processing_time_ms: float

class FaceMeshDetector:
    """
    Detects facial landmarks using MediaPipe FaceMesh.
    Provides methods to process frames, draw landmarks,
    and draw a bounding box around the face.
    
    Optionally uses a custom landmark configuration; if none is provided,
    defaults to the MediaPipe configuration in constants.
    """
    def __init__(self,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 max_num_faces: int = 1,
                 landmark_config: Optional[LandmarkIndices] = None,
                 log_interval: int = 100):
        """
        Args:
            refine_landmarks: Whether to refine landmarks (improves accuracy).
            min_detection_confidence: Confidence threshold for detection.
            min_tracking_confidence: Confidence threshold for tracking.
            max_num_faces: Maximum number of faces to detect.
            landmark_config: Optional custom landmark indices configuration.
                            If not provided, uses DEFAULT.
            log_interval: Number of frames between logging detection stats.
        """
        self._validate_confidence(min_detection_confidence, "min_detection_confidence")
        self._validate_confidence(min_tracking_confidence, "min_tracking_confidence")
        self._validate_max_faces(max_num_faces)
        
        self.landmark_config = landmark_config if landmark_config is not None else DEFAULT
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self._face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False
        )
        
        self._last_detection: Optional[FaceDetectionResult] = None
        self._frame_count = 0
        self._detection_count = 0
        self._log_interval = log_interval
        
        logger.info(f"FaceMeshDetector initialized with max_num_faces={max_num_faces}, "
                    f"refine_landmarks={refine_landmarks}")

    @staticmethod
    def _validate_confidence(value: float, name: str) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number; got {type(value).__name__}.")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be between 0.0 and 1.0; got {value}.")

    @staticmethod
    def _validate_max_faces(value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"max_num_faces must be an integer; got {type(value).__name__}.")
        if value <= 0:
            raise ValueError(f"max_num_faces must be positive; got {value}.")

    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:
        return (isinstance(frame, np.ndarray) and 
                frame.size > 0 and 
                len(frame.shape) == 3 and 
                frame.shape[2] == 3)

    def process_frame(self, frame: NDArray[np.uint8]) -> Optional[FaceDetectionResult]:
        """
        Processes the given frame to extract facial landmarks and metadata.
        
        Returns a FaceDetectionResult or None if no face is detected.
        """
        self._frame_count += 1
        
        if not self._is_valid_frame(frame):
            logger.warning(f"Received an invalid frame (shape: {getattr(frame, 'shape', 'unknown')})")
            return None
            
        try:
            start_tick = cv2.getTickCount()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_frame)
            end_tick = cv2.getTickCount()
            processing_time_ms = (end_tick - start_tick) * 1000 / cv2.getTickFrequency()
            
            if not results.multi_face_landmarks:
                self._log_detection_rate()
                return None
                
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # Vectorized conversion from normalized to pixel coordinates.
            lm_x = np.array([lm.x for lm in face_landmarks]) * w
            lm_y = np.array([lm.y for lm in face_landmarks]) * h
            landmarks_pixels = np.stack((lm_x, lm_y), axis=-1).astype(np.float32)
            
            # Calculate bounding box
            x, y, box_w, box_h = cv2.boundingRect(landmarks_pixels.astype(np.int32))
            
            # Dynamic confidence calculation based on detection rate.
            detection_rate = (self._detection_count / self._frame_count) if self._frame_count else 1.0
            confidence = min(1.0, detection_rate * 1.2)
            
            result = FaceDetectionResult(
                landmarks=landmarks_pixels,
                bounding_box=(x, y, box_w, box_h),
                confidence=confidence,
                processing_time_ms=processing_time_ms
            )
            
            self._last_detection = result
            self._detection_count += 1
            logger.info(f"Face detection rate: {detection_rate:.1f}% ({self._detection_count}/{self._frame_count} frames)")
            return result
        except Exception as e:
            logger.error("Error processing frame: %s", e, exc_info=True)
            return None

    def _log_detection_rate(self) -> None:
        if self._frame_count % self._log_interval == 0:
            detection_rate = (self._detection_count / self._frame_count) * 100
            logger.info(f"Face detection rate: {detection_rate:.1f}% "
                        f"({self._detection_count}/{self._frame_count} frames)")

    def draw_landmarks(self,
                       frame: NDArray[np.uint8],
                       detection_result: Optional[FaceDetectionResult],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       radius: int = 2,
                       draw_connections: bool = False) -> None:
        """
        Draws facial landmarks on the frame.
        """
        if detection_result is None or detection_result.landmarks.size == 0:
            return
            
        landmarks = detection_result.landmarks
        
        if draw_connections:
            h, w = frame.shape[:2]
            landmark_proto = self.mp_face_mesh.LandmarkList()
            for pt in landmarks:
                landmark = landmark_proto.landmark.add()
                landmark.x = pt[0] / w
                landmark.y = pt[1] / h
                landmark.z = 0
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_proto,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        else:
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    def draw_face_bounding_box(self,
                               frame: NDArray[np.uint8],
                               detection_result: Optional[FaceDetectionResult],
                               box_color: Tuple[int, int, int] = (0, 0, 255),
                               text_color: Tuple[int, int, int] = (255, 255, 255),
                               show_confidence: bool = True) -> None:
        """
        Draws a bounding box around the face and optionally displays the confidence score.
        """
        if detection_result is None:
            return
        x, y, w, h = detection_result.bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        label = f"Face {detection_result.confidence:.2f}" if show_confidence else "Face"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y - 10 if y - 10 > label_size[1] else y + 10
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, text_color, 1, cv2.LINE_AA)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._face_mesh.close()
        if exc_type is not None:
            logger.error("Exception in FaceMeshDetector context: %s", exc_value, exc_info=True)
