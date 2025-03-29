# head_pose.py
import cv2
import numpy as np
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
        smooth_factor: float = 0.2,
        stability_window: int = 10,  # Number of frames to check for stability
        stability_threshold: float = 5.0  # Maximum allowed variance for stability
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
        """Initialize camera matrix if not provided"""
        focal_length = frame_size[1]
        center = (frame_size[1]/2, frame_size[0]/2)
        self._camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

    def _solve_pnp(self, image_points: NDArray[np.float32]) -> Optional[Tuple]:
        """Core PnP solution with error handling"""
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                self._3D_REF_POINTS,
                image_points,
                self._camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                confidence=0.99,
                reprojectionError=5.0
            )
            return rvec, tvec, inliers
        except Exception as e:
            print(f"PnP solving error: {str(e)}")
            return None

    def _calculate_reprojection_error(
        self,
        rvec: NDArray,
        tvec: NDArray,
        image_points: NDArray
    ) -> float:
        """Calculate mean reprojection error"""
        projected, _ = cv2.projectPoints(
            self._3D_REF_POINTS,
            rvec,
            tvec,
            self._camera_matrix,
            self.dist_coeffs
        )
        return float(np.linalg.norm(projected.squeeze() - image_points))

    def _convert_to_euler(self, rvec: NDArray) -> Tuple[float, float, float]:
        """Convert rotation vector to Euler angles (pitch, yaw, roll)"""
        rotation_mat, _ = cv2.Rodrigues(rvec)
        
        # Validate rotation matrix
        if not np.isfinite(rotation_mat).all() or np.abs(rotation_mat[2, 0]) >= 1.0:
            print("Invalid rotation matrix detected. Returning fallback angles.")
            return 0.0, 0.0, 0.0
        
        euler_angles = cv2.RQDecomp3x3(rotation_mat)[0]
        
        # Clamp angles to avoid extreme values
        return tuple(max(min(angle, 90.0), -90.0) for angle in euler_angles)

    def _smooth_euler_angles(self, new_angles: NDArray):
        """Apply exponential smoothing to Euler angles"""
        if not np.isfinite(new_angles).all():
            print("Invalid Euler angles detected. Skipping smoothing.")
            return
        self._smoothed_euler = (self.smooth_factor * new_angles + 
                              (1 - self.smooth_factor) * self._smoothed_euler)

    def _is_pose_stable(self) -> bool:
        """
        Check if the head pose values are stable over the recent frames.
        
        Returns:
            bool: True if the variance of pitch, yaw, and roll is below the threshold.
        """
        if len(self._pose_history) < self.stability_window:
            return True  # Not enough data to determine stability

        # Calculate variance for pitch, yaw, and roll
        pose_array = np.array(self._pose_history)
        variances = np.var(pose_array, axis=0)

        # Check if all variances are below the threshold
        is_stable = np.all(variances < self.stability_threshold)
        if not is_stable:
            print(f"HeadPoseEstimator: Unstable pose detected. Variances: {variances}")
        return is_stable

    @property
    def confidence(self) -> float:
        """Calculate confidence score based on recent errors"""
        if not self._reprojection_errors:
            return 0.0
        avg_error = np.mean(self._reprojection_errors)
        return np.exp(-avg_error / 100.0)

    def process_frame(self, frame: NDArray[np.uint8]) -> HeadPoseResult:
        """Process a frame and return head pose results"""
        landmarks = self.detector.process_frame(frame)
        if landmarks is None:
            print("HeadPoseEstimator: No landmarks detected. Returning fallback result.")
            return self._fallback_result(frame)

        # Initialize camera matrix if needed
        if self._camera_matrix is None:
            self._initialize_camera_matrix(frame.shape[:2])

        # Get 2D image points from landmarks
        image_points = landmarks[self._LANDMARK_INDICES]

        # Solve PnP
        pnp_result = self._solve_pnp(image_points)
        if pnp_result is None:
            print("HeadPoseEstimator: PnP solution not found. Returning fallback result.")
            return self._fallback_result(frame)

        rvec, tvec, inliers = pnp_result
        tvec = tvec.squeeze()

        # Calculate metrics
        reproj_error = self._calculate_reprojection_error(rvec, tvec, image_points)
        self._reprojection_errors.append(reproj_error)
        
        # Convert to Euler angles and smooth
        euler_angles = self._convert_to_euler(rvec)
        self._smooth_euler_angles(np.array(euler_angles))

        result = HeadPoseResult(
            rvec=rvec,
            tvec=tvec,
            euler_angles=tuple(self._smoothed_euler.tolist()),
            landmarks=landmarks,
            confidence=self.confidence
        )

        if result.confidence < 0.4:
            print(f"HeadPoseEstimator: Low confidence ({result.confidence:.2f}). Returning fallback result.")
            return self._fallback_result(frame)

        # Add the current pose to the history
        self._pose_history.append(result.euler_angles)

        # Validate stability
        if not self._is_pose_stable():
            print("HeadPoseEstimator: Pose is unstable. Returning fallback result.")
            return self._fallback_result(frame)

        self._last_valid_result = result  # Update last valid result
        return result

    def _fallback_result(self, frame: NDArray[np.uint8]) -> HeadPoseResult:
        """Return the last valid result or a default result"""
        if self._last_valid_result:
            print("Using last valid head pose result.")
            return self._last_valid_result
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
            if estimator._camera_matrix is not None and result.confidence > 0.4:
                frame = HeadPoseEstimator.draw_pose_axes(
                    frame, result, estimator._camera_matrix
                )

            # Display Euler angles
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