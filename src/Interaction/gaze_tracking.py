# gaze_tracking.py
import cv2
import numpy as np
import time
from scipy.spatial import Delaunay
from typing import NamedTuple, Optional, List, Dict, Deque, Tuple
from collections import deque
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector
from Interaction.eye_pupil_extract import EyeRegionExtractor
from Interaction.eye_blink import EyeBlinkDetector
from Interaction.constants import FaceLandmarks, DEFAULT  # Import DEFAULT explicitly
from Interaction.head_pose import HeadPoseEstimator  # Import HeadPoseEstimator

# Enhanced configuration
class GazeConfig(NamedTuple):
    calibration_points: List[Tuple[float, float]] = [(0.2, 0.2), (0.8, 0.2), 
                                                    (0.5, 0.5), (0.2, 0.8), 
                                                    (0.8, 0.8)]
    calibration_duration: float = 3.0
    fixation_threshold: float = 0.1  # Degrees of visual angle
    fixation_duration: float = 0.3    # Seconds
    blink_threshold: float = 0.15     # EAR threshold  # Ensure this is defined here
    confidence_weights: Dict[str, float] = {
        'pupil_quality': 0.4,
        'head_stability': 0.3,
        'calibration_match': 0.3
    }

class CalibrationPoint(NamedTuple):
    screen_pos: Tuple[float, float]
    gaze_vectors: List[Tuple[float, float, float]]
    head_poses: List[Tuple[float, float, float]]

class GazeResult(NamedTuple):
    screen_coord: Optional[Tuple[float, float]]
    gaze_3d: Optional[Tuple[float, float, float]]
    fixation: Optional[Tuple[float, float]]
    confidence: float
    is_blinking: bool
    calibration_progress: float
    frame: NDArray[np.uint8]

class GazeCalibrator:
    """Handles calibration process and coordinate transformation"""
    def __init__(self, config: GazeConfig):
        self.config = config
        self.calibration_data: List[CalibrationPoint] = []
        self.calibration_model: Optional[Dict] = None
        self.current_point = 0
        self.calibration_start_time = 0.0

    def start_calibration(self, screen_resolution: Tuple[int, int]):
        self.screen_res = screen_resolution
        self.calibration_data = []
        self.current_point = 0
        self.calibration_start_time = time.time()

    def add_calibration_sample(self, gaze_vector: Tuple[float, float, float],
                             head_pose: Tuple[float, float, float]):
        if self.current_point >= len(self.config.calibration_points):
            return

        current_pos = self.config.calibration_points[self.current_point]
        screen_pos = (current_pos[0] * self.screen_res[0],
                     current_pos[1] * self.screen_res[1])

        if len(self.calibration_data) <= self.current_point:
            self.calibration_data.append(CalibrationPoint(
                screen_pos=screen_pos,
                gaze_vectors=[],
                head_poses=[]
            ))
            
        self.calibration_data[self.current_point].gaze_vectors.append(gaze_vector)
        self.calibration_data[self.current_point].head_poses.append(head_pose)

    def update_calibration_progress(self):
        elapsed = time.time() - self.calibration_start_time
        if elapsed > self.config.calibration_duration:
            if self.current_point < len(self.config.calibration_points) - 1:
                self.current_point += 1
                self.calibration_start_time = time.time()
            else:
                self._build_calibration_model()

    def _build_calibration_model(self):
        """Create screen mapping using linear regression"""
        if not self.calibration_data:
            print("GazeCalibrator: No calibration data available.")
            return

        # Prepare data for regression
        screen_coords = []
        gaze_vectors = []
        for point in self.calibration_data:
            screen_coords.extend([point.screen_pos] * len(point.gaze_vectors))
            gaze_vectors.extend(point.gaze_vectors)

        screen_coords = np.array(screen_coords)
        gaze_vectors = np.array(gaze_vectors)

        # Fit linear regression model
        try:
            self.calibration_model = {}
            for i in range(2):  # x and y screen coordinates
                coeffs = np.linalg.lstsq(gaze_vectors, screen_coords[:, i], rcond=None)[0]
                self.calibration_model[f"axis_{i}"] = coeffs
            print("GazeCalibrator: Calibration model built successfully.")
        except Exception as e:
            print(f"GazeCalibrator: Error building calibration model - {e}")

    def map_to_screen(self, gaze_vector: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """Map a 3D gaze vector to screen coordinates using the calibration model"""
        if not self.calibration_model:
            print("GazeCalibrator: Calibration model not available.")
            return None

        try:
            gaze_vector = np.array(gaze_vector)
            x = np.dot(gaze_vector, self.calibration_model["axis_0"])
            y = np.dot(gaze_vector, self.calibration_model["axis_1"])
            return x, y
        except Exception as e:
            print(f"GazeCalibrator: Error mapping gaze vector to screen - {e}")
            return None

    def get_current_target(self) -> Tuple[float, float]:
        return self.config.calibration_points[self.current_point]

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_model is not None

class GazeTracker:
    def __init__(self, face_detector, eye_extractor, blink_detector, config: GazeConfig, landmarks_config=DEFAULT):
        self.detector = face_detector
        self.eye_extractor = eye_extractor
        self.blink_detector = blink_detector
        self.config = config
        self.landmarks_config = landmarks_config
        self.calibrator = GazeCalibrator(config)
        self.head_pose_estimator = HeadPoseEstimator(face_detector)  # Initialize HeadPoseEstimator
        
        # State tracking
        self.gaze_history: Deque[Tuple[float, float]] = deque(maxlen=30)
        self.fixation_start: Optional[float] = None
        self.head_pose_history: Deque[Tuple[float, float, float]] = deque(maxlen=10)

    def _estimate_3d_gaze(self, eye_data, head_pose):
        """Convert 2D gaze vector to 3D using head pose"""
        # Implementation using ray casting and head rotation
        pass

    def _estimate_gaze_vector(self, eye_regions: Tuple[EyeRegionExtractor, EyeRegionExtractor]) -> Tuple[Tuple[float, float, float], float]:
        """Estimate the 3D gaze vector using eye regions"""
        if not eye_regions[0].is_valid or not eye_regions[1].is_valid:
            print("GazeTracker: Invalid eye regions for gaze estimation.")
            return (0.0, 0.0, 0.0), 0.0

        # Simplified gaze vector estimation (e.g., midpoint of eye centers)
        left_eye_center = np.mean(eye_regions[0].landmarks, axis=0)
        right_eye_center = np.mean(eye_regions[1].landmarks, axis=0)
        gaze_vector = np.array([(left_eye_center[0] + right_eye_center[0]) / 2,
                                (left_eye_center[1] + right_eye_center[1]) / 2,
                                1.0])  # Assume a fixed depth for simplicity
        return gaze_vector, 1.0  # Return a dummy confidence for now

    def _estimate_head_pose(self, landmarks: NDArray[np.float32]) -> Optional[Tuple[float, float, float]]:
        """Estimate head pose using HeadPoseEstimator."""
        if landmarks is None:
            return None
        head_pose_result = self.head_pose_estimator.process_frame(landmarks)
        if head_pose_result.confidence < 0.4:  # Ensure confidence is sufficient
            return None
        return head_pose_result.euler_angles  # Return pitch, yaw, roll

    def _detect_fixation(self):
        """Calculate fixation using dispersion threshold"""
        if len(self.gaze_history) < 10:
            return None

        points = np.array(self.gaze_history)
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        dispersion = np.mean(distances)

        if dispersion < self.config.fixation_threshold:
            if self.fixation_start is None:
                self.fixation_start = time.time()
            elif time.time() - self.fixation_start > self.config.fixation_duration:
                return centroid
        else:
            self.fixation_start = None
        return None

    def _calculate_confidence(self, pupil_quality, head_stability):
        """Composite confidence score"""
        calib_match = self.calibrator.calibration_quality if self.calibrator.is_calibrated else 1.0
        return (
            self.config.confidence_weights['pupil_quality'] * pupil_quality +
            self.config.confidence_weights['head_stability'] * head_stability +
            self.config.confidence_weights['calibration_match'] * calib_match
        )

    def process_frame(self, frame: NDArray[np.uint8]) -> GazeResult:
        # Detect blinks first
        blink_result = self.blink_detector.process_frame(frame)
        is_blinking = blink_result['ear'] < self.config.blink_threshold

        if is_blinking:
            return GazeResult(
                screen_coord=None,
                gaze_3d=None,
                fixation=None,
                confidence=0.0,
                is_blinking=True,
                calibration_progress=self.calibrator.current_point / len(self.config.calibration_points),
                frame=frame
            )

        # Gaze estimation pipeline
        landmarks = self.detector.process_frame(frame)
        if landmarks is None:
            print("GazeTracker: No landmarks detected. Skipping gaze tracking.")
            return GazeResult(
                screen_coord=None,
                gaze_3d=None,
                fixation=None,
                confidence=0.0,
                is_blinking=False,
                calibration_progress=self.calibrator.current_point / len(self.config.calibration_points),
                frame=frame
            )

        head_pose = self._estimate_head_pose(landmarks)
        if head_pose is None:
            print("GazeTracker: Head pose estimation failed. Skipping gaze tracking.")
            return GazeResult(
                screen_coord=None,
                gaze_3d=None,
                fixation=None,
                confidence=0.0,
                is_blinking=False,
                calibration_progress=self.calibrator.current_point / len(self.config.calibration_points),
                frame=frame
            )

        eye_regions = self.eye_extractor.extract_eye_regions(frame)
        if not eye_regions[0].is_valid or not eye_regions[1].is_valid:
            print("GazeTracker: Invalid eye regions. Skipping gaze tracking.")
            return GazeResult(
                screen_coord=None,
                gaze_3d=None,
                fixation=None,
                confidence=0.0,
                is_blinking=False,
                calibration_progress=self.calibrator.current_point / len(self.config.calibration_points),
                frame=frame
            )

        # Calibration prompt
        if not self.calibrator.is_calibrated:
            target = self.calibrator.get_current_target()
            cv2.circle(frame, (int(target[0]), int(target[1])), 20, (0, 255, 255), 3)
            cv2.putText(frame, "Look here for calibration!", (int(target[0]) - 60, int(target[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            gaze_vector, _ = self._estimate_gaze_vector(eye_regions)  # Calculate gaze_vector
            self.calibrator.add_calibration_sample(gaze_vector, head_pose)
            self.calibrator.update_calibration_progress()

        gaze_2d, pupil_quality = self._estimate_gaze_vector(eye_regions)
        gaze_3d = self._estimate_3d_gaze(gaze_2d, head_pose)
        
        # Store historical data
        self.gaze_history.append(gaze_2d)
        self.head_pose_history.append(head_pose)

        # Calculate outputs
        screen_coord = self.calibrator.map_to_screen(gaze_3d) if self.calibrator.is_calibrated else None
        fixation = self._detect_fixation()
        head_stability = np.mean(np.std(self.head_pose_history, axis=0))
        confidence = self._calculate_confidence(pupil_quality, head_stability)

        return GazeResult(
            screen_coord=screen_coord,
            gaze_3d=gaze_3d,
            fixation=fixation,
            confidence=confidence,
            is_blinking=False,
            calibration_progress=self.calibrator.current_point / len(self.config.calibration_points),
            frame=self._draw_ui(frame, gaze_3d, screen_coord, fixation, confidence)
        )

    def _draw_ui(self, frame, gaze_3d, screen_coord, fixation, confidence):
        """Draw enhanced visualization"""
        # Draw 3D gaze vector
        if gaze_3d:
            cv2.putText(frame, f"3D Gaze: {gaze_3d}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw screen coordinates
        if screen_coord:
            cv2.circle(frame, (int(screen_coord[0]), int(screen_coord[1])),
                      5, (0, 0, 255), -1)
        
        # Draw fixation
        if fixation:
            cv2.putText(frame, "FIXATION", (frame.shape[1]//2 - 50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw confidence meter
        cv2.rectangle(frame, (10, frame.shape[0]-20), (110, frame.shape[0]-10),
                      (255,255,255), 1)
        cv2.rectangle(frame, (10, frame.shape[0]-20), 
                     (10 + int(100 * confidence), frame.shape[0]-10),
                     (0,255,0), -1)
        
        # Calibration UI
        if not self.calibrator.is_calibrated:
            target = self.calibrator.get_current_target()
            cv2.circle(frame, (int(target[0]), int(target[1])),
                      20, (0,255,255), 3)
            cv2.putText(frame, "Look here!", (int(target[0])-60, int(target[1])-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        return frame

    def start_calibration_sequence(self, screen_resolution: Tuple[int, int]):
        self.calibrator.start_calibration(screen_resolution)

# Usage example
if __name__ == "__main__":
    config = GazeConfig()
    with FaceMeshDetector() as detector, \
         EyeRegionExtractor(detector) as eye_extractor, \
         EyeBlinkDetector(detector) as blink_detector:

        tracker = GazeTracker(detector, eye_extractor, blink_detector, config)
        tracker.start_calibration_sequence((1920, 1080))

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = tracker.process_frame(frame)
            
            if tracker.calibrator.is_calibrated:
                # Use screen coordinates for applications
                print(f"Screen: {result.screen_coord}, Confidence: {result.confidence:.2f}")
            
            cv2.imshow("Gaze Tracking", result.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()