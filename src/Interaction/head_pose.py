# head_pose.py
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from collections import deque
from typing import NamedTuple, Optional, Tuple, Deque
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector

class HeadPoseResult(NamedTuple):
    rvec: NDArray[np.float32]          # Rotation vector (3x1)
    tvec: NDArray[np.float32]          # Translation vector (3x1)
    euler_angles: Tuple[float, float, float]  # (pitch, yaw, roll) in degrees
    landmarks: Optional[NDArray[np.float32]]
    confidence: float

class HeadPoseEstimator:
    """
    3D Head Pose Estimation using Perspective-n-Point (PnP) algorithm
    with proper rotation handling and visualization
    
    Features:
    - Accurate rotation vector preservation from solvePnP
    - Euler angles conversion for human-readable output
    - Confidence scoring based on reprojection error
    - Exponential smoothing for stable outputs
    - 3D axis visualization with proper rotation handling
    """

    # 3D reference points (world coordinates in meters)
    _3D_REF_POINTS = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -0.35, -0.1),        # Chin
        (-0.225, 0.17, -0.12),     # Left eye corner
        (0.225, 0.17, -0.12),      # Right eye corner
        (-0.15, -0.15, -0.12),     # Left mouth corner
        (0.15, -0.15, -0.12)       # Right mouth corner
    ], dtype=np.float32)

    # Corresponding MediaPipe Face Mesh landmark indices
    _LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]

    def __init__(
        self,
        face_detector: FaceMeshDetector,
        camera_matrix: Optional[NDArray[np.float32]] = None,
        dist_coeffs: Optional[NDArray[np.float32]] = None,
        smooth_factor: float = 0.5,  # Increased smoothing factor
        stability_window: int = 20,  # Increased stability window
        stability_threshold: float = 2.0  # Reduced stability threshold
    ):
        """
        Args:
            face_detector: Initialized face detector
            camera_matrix: 3x3 camera matrix
            dist_coeffs: Distortion coefficients
            smooth_factor: Smoothing factor (0-1) for Euler angles
            stability_window: Number of frames to check for stability
            stability_threshold: Maximum allowed variance for stability
        """
        self.detector = face_detector
        self.smooth_factor = smooth_factor
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))
        
        # State variables
        self._camera_matrix = camera_matrix
        self._smoothed_euler = np.zeros(3, dtype=np.float32)
        self._reprojection_errors = deque(maxlen=10)
        self._last_valid_result: Optional[HeadPoseResult] = None  # Store last valid result
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self._pose_history = deque(maxlen=stability_window)  # Store recent head poses

    def _initialize_camera_matrix(self, frame_size: Tuple[int, int]):
        """Initialize camera matrix with dynamic focal length."""
        focal_length = max(frame_size)  # Use the larger dimension for better accuracy
        center = (frame_size[1] / 2, frame_size[0] / 2)
        self._camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

    def _solve_pnp(self, image_points: NDArray[np.float32]) -> Optional[Tuple]:
        """Core PnP solution with stricter constraints."""
        try:
            success, rvec, tvec = cv2.solvePnP(
                self._3D_REF_POINTS,
                image_points,
                self._camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                print("HeadPoseEstimator: PnP solution failed.")
                return None
            return rvec, tvec
        except Exception as e:
            print(f"HeadPoseEstimator: Error solving PnP - {e}")
            return None

    def _validate_landmarks(self, landmarks: NDArray[np.float32]) -> bool:
        """Validate that all required landmarks are detected and within frame boundaries."""
        if landmarks is None or len(landmarks) < len(self._LANDMARK_INDICES):
            print("HeadPoseEstimator: Missing or invalid landmarks.")
            return False
        if not np.isfinite(landmarks).all():
            print("HeadPoseEstimator: Landmarks contain invalid values.")
            return False
        return True

    def _calculate_reprojection_error(
        self,
        rvec: NDArray,
        tvec: NDArray,
        image_points: NDArray
    ) -> float:
        """Calculate mean reprojection error."""
        try:
            projected, _ = cv2.projectPoints(
                self._3D_REF_POINTS,
                rvec,
                tvec,
                self._camera_matrix,
                self.dist_coeffs
            )
            error = np.linalg.norm(projected.squeeze() - image_points, axis=1).mean()
            return error
        except Exception as e:
            print(f"HeadPoseEstimator: Error calculating reprojection error - {e}")
            return float('inf')

    def _convert_to_euler(self, rvec: NDArray) -> Tuple[float, float, float]:
        """Convert rotation vector to Euler angles (pitch, yaw, roll)."""
        try:
            rotation_mat, _ = cv2.Rodrigues(rvec)
            if not np.isfinite(rotation_mat).all():
                raise ValueError("Invalid rotation matrix detected.")

            euler_angles = cv2.RQDecomp3x3(rotation_mat)[0]
            return tuple(max(min(angle, 90.0), -90.0) for angle in euler_angles)
        except Exception as e:
            print(f"HeadPoseEstimator: Error converting to Euler angles - {e}")
            return 0.0, 0.0, 0.0

    def _smooth_euler_angles(self, new_angles: NDArray):
        """Apply exponential smoothing to Euler angles"""
        if not np.isfinite(new_angles).all():
            print("Invalid Euler angles detected. Skipping smoothing.")
            return
        self._smoothed_euler = (self.smooth_factor * new_angles + 
                              (1 - self.smooth_factor) * self._smoothed_euler)

    def _is_pose_stable(self) -> bool:
        """Check if the head pose values are stable over the recent frames."""
        if len(self._pose_history) < self.stability_window:
            return True  # Not enough data to determine stability

        pose_array = np.array(self._pose_history)
        variances = np.var(pose_array, axis=0)
        is_stable = np.all(variances < self.stability_threshold)
        if not is_stable:
            print(f"HeadPoseEstimator: Unstable pose detected. Variances: {variances}")
            # Fallback: Use the last valid stable pose if available
            if self._last_valid_result:
                self._pose_history.clear()  # Reset history to avoid cascading instability
                return True
        return is_stable

    def _validate_physical_constraints(self, euler_angles: Tuple[float, float, float]) -> bool:
        """Validate that the head pose is within realistic physical constraints."""
        pitch, yaw, roll = euler_angles
        if not (-90.0 <= pitch <= 90.0 and -90.0 <= yaw <= 90.0 and -45.0 <= roll <= 45.0):
            print(f"HeadPoseEstimator: Pose out of physical constraints. Pitch={pitch}, Yaw={yaw}, Roll={roll}")
            return False
        return True

    @property
    def confidence(self) -> float:
        """Calculate confidence score based on recent errors"""
        if not self._reprojection_errors:
            return 0.0
        avg_error = np.mean(self._reprojection_errors)
        return np.exp(-avg_error / 100.0)

    @staticmethod
    def draw_pose_point(
        frame: NDArray[np.uint8],
        result: HeadPoseResult,
        camera_matrix: NDArray[np.float32]
    ) -> None:
        """Draw the head pose point (nose tip) on the frame."""
        if result.confidence < 0.4 or result.landmarks is None:
            return

        # Get the nose tip point
        nose_tip = tuple(result.landmarks[1].astype(int))
        cv2.circle(frame, nose_tip, 5, (0, 0, 255), -1)  # Red dot for the nose tip
        cv2.putText(frame, "Nose Tip", (nose_tip[0] + 10, nose_tip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def draw_face_bounding_box(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> None:
        """
        Draw a bounding box around the face.
        """
        x, y, w, h = cv2.boundingRect(landmarks.astype(np.int32))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for the face
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def process_frame(self, frame: NDArray[np.uint8]) -> Optional[HeadPoseResult]:
        """Process a frame and return head pose results."""
        # Log frame dimensions for debugging
        print(f"Processing frame of shape: {frame.shape}")

        # Preprocess frame if needed (e.g., resizing or normalization)
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, 720))
            print("Frame resized to 1280x720 for consistent processing.")

        # Validate landmarks
        landmarks = self.detector.process_frame(frame)
        if not self._validate_landmarks(landmarks):
            return self._fallback_result(frame)

        # Initialize camera matrix if needed
        if self._camera_matrix is None:
            self._initialize_camera_matrix(frame.shape[:2])

        # Get 2D image points from landmarks
        image_points = landmarks[self._LANDMARK_INDICES]

        # Solve PnP
        pnp_result = self._solve_pnp(image_points)
        if pnp_result is None:
            print("HeadPoseEstimator: PnP solution not found.")
            return self._fallback_result(frame)

        rvec, tvec = pnp_result

        # Convert to Euler angles
        euler_angles = self._convert_to_euler(rvec)
        if not self._validate_physical_constraints(euler_angles):
            print(f"HeadPoseEstimator: Pose out of physical constraints. Pitch={euler_angles[0]:.1f}, "
                  f"Yaw={euler_angles[1]:.1f}, Roll={euler_angles[2]:.1f}")
            return self._fallback_result(frame)

        # Smooth Euler angles
        self._smooth_euler_angles(np.array(euler_angles))

        # Update last valid result
        self._last_valid_result = HeadPoseResult(
            rvec=rvec,
            tvec=tvec,
            euler_angles=tuple(self._smoothed_euler.tolist()),
            landmarks=landmarks,
            confidence=1.0  # Set confidence to 1.0 for a single valid candidate
        )

        # Draw face bounding box
        if landmarks is not None:
            self.draw_face_bounding_box(frame, landmarks)

        # Draw head pose point
        if self._last_valid_result:
            self.draw_pose_point(frame, self._last_valid_result, self._camera_matrix)

        return self._last_valid_result

    def _fallback_result(self, frame: NDArray[np.uint8]) -> HeadPoseResult:
        """Return the last valid result or a default result."""
        if self._last_valid_result:
            print("Using last valid head pose result.")
            return self._last_valid_result
        print("No valid pose available. Returning default pose.")
        return HeadPoseResult(
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            euler_angles=(0.0, 0.0, 0.0),
            landmarks=None,
            confidence=0.0
        )

    @staticmethod
    def draw_pose_axes(
        frame: NDArray[np.uint8],
        result: HeadPoseResult,
        camera_matrix: NDArray[np.float32],
        length: float = 0.1
    ) -> NDArray[np.uint8]:
        """Draw 3D pose axes using original rotation vector"""
        if result.confidence < 0.4 or result.landmarks is None:
            return frame

        # Define axis points in 3D space
        axis = np.float32([[length,0,0], [0,length,0], [0,0,length]])

        # Project 3D points to 2D image plane
        imgpts, _ = cv2.projectPoints(
            axis,
            result.rvec,
            result.tvec,
            camera_matrix,
            np.zeros(4,1)
        )

        # Get nose base point
        nose_point = tuple(result.landmarks[1].astype(int))

        # Draw axes
        colors = [(0,0,255), (0,255,0), (255,0,0)]  # BGR
        for i, color in enumerate(colors):
            end_point = tuple(imgpts[i].ravel().astype(int))
            cv2.line(frame, nose_point, end_point, color, 3)

        return frame

def main():
    """Example usage with webcam feed"""
    with FaceMeshDetector() as detector:
        estimator = HeadPoseEstimator(detector)
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = estimator.process_frame(frame)
            
            # Draw visualization if confidence is high
            if estimator._camera_matrix is not None and result and result.confidence > 0.4:
                frame = HeadPoseEstimator.draw_pose_axes(
                    frame, result, estimator._camera_matrix
                )

            # Display Euler angles
            if result:
                pitch, yaw, roll = result.euler_angles
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Confidence: {result.confidence:.2f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Head Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()