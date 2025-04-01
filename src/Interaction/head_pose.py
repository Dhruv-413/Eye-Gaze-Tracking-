# head_pose.py
import cv2
import numpy as np
import logging
from collections import deque
from typing import NamedTuple, Optional, Tuple, Deque
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HeadPoseResult(NamedTuple):
    rvec: NDArray[np.float32]
    tvec: NDArray[np.float32]
    euler_angles: Tuple[float, float, float]
    landmarks: Optional[NDArray[np.float32]]
    confidence: float

class HeadPoseEstimator:
    """
    3D Head Pose Estimation using PnP with exponential smoothing and stability checks.
    """
    _3D_REF_POINTS = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -0.35, -0.1),
        (-0.225, 0.17, -0.12),
        (0.225, 0.17, -0.12),
        (-0.15, -0.15, -0.12),
        (0.15, -0.15, -0.12)
    ], dtype=np.float32)
    _LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]

    def __init__(
        self,
        face_detector: FaceMeshDetector,
        camera_matrix: Optional[NDArray[np.float32]] = None,
        dist_coeffs: Optional[NDArray[np.float32]] = None,
        smooth_factor: float = 0.5,
        stability_window: int = 20,
        stability_threshold: float = 2.0
    ):
        self.detector = face_detector
        self.smooth_factor = smooth_factor
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4,1), dtype=np.float32)
        self._camera_matrix = camera_matrix
        self._smoothed_euler = np.zeros(3, dtype=np.float32)
        self._reprojection_errors: Deque[float] = deque(maxlen=10)
        self._pose_history: Deque[NDArray[np.float32]] = deque(maxlen=stability_window)
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self._last_valid_result: Optional[HeadPoseResult] = None

    def _initialize_camera_matrix(self, frame_size: Tuple[int,int]) -> None:
        focal_length = max(frame_size)
        center = (frame_size[1]/2, frame_size[0]/2)
        self._camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0,0,1]
        ], dtype=np.float32)
        logger.info("Initialized camera matrix: %s", self._camera_matrix)

    def _solve_pnp(self, image_points: NDArray[np.float32]) -> Optional[Tuple[NDArray[np.float32], NDArray[np.float32]]]:
        try:
            success, rvec, tvec = cv2.solvePnP(
                self._3D_REF_POINTS,
                image_points,
                self._camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                logger.warning("PnP solution failed.")
                return None
            return rvec, tvec
        except Exception as e:
            logger.error("Error solving PnP: %s", e, exc_info=True)
            return None

    def _validate_landmarks(self, landmarks: NDArray[np.float32]) -> bool:
        if landmarks is None or len(landmarks) < len(self._LANDMARK_INDICES):
            logger.warning("Missing or invalid landmarks.")
            return False
        if not np.isfinite(landmarks).all():
            logger.warning("Landmarks contain invalid values.")
            return False
        return True

    def _calculate_reprojection_error(self, rvec: NDArray, tvec: NDArray, image_points: NDArray) -> float:
        try:
            projected, _ = cv2.projectPoints(self._3D_REF_POINTS, rvec, tvec, self._camera_matrix, self.dist_coeffs)
            error = np.linalg.norm(projected.squeeze() - image_points, axis=1).mean()
            return error
        except Exception as e:
            logger.error("Error calculating reprojection error: %s", e, exc_info=True)
            return float('inf')

    def _convert_to_euler(self, rvec: NDArray) -> Tuple[float, float, float]:
        try:
            rotation_mat, _ = cv2.Rodrigues(rvec)
            if not np.isfinite(rotation_mat).all():
                raise ValueError("Invalid rotation matrix.")
            euler_angles = cv2.RQDecomp3x3(rotation_mat)[0]
            return tuple(max(min(angle, 90.0), -90.0) for angle in euler_angles)
        except Exception as e:
            logger.error("Error converting to Euler angles: %s", e, exc_info=True)
            return (0.0, 0.0, 0.0)

    def _smooth_euler_angles(self, new_angles: NDArray[np.float32]) -> None:
        if not np.isfinite(new_angles).all():
            logger.warning("Invalid Euler angles; skipping smoothing.")
            return
        self._smoothed_euler = self.smooth_factor * new_angles + (1 - self.smooth_factor) * self._smoothed_euler
        self._pose_history.append(self._smoothed_euler.copy())

    def _is_pose_stable(self) -> bool:
        if len(self._pose_history) < self.stability_window:
            return True
        pose_array = np.array(self._pose_history)
        variances = np.var(pose_array, axis=0)
        if np.all(variances < self.stability_threshold):
            return True
        logger.info("Unstable pose detected; variances: %s", variances)
        return False

    def _validate_physical_constraints(self, euler_angles: Tuple[float, float, float]) -> bool:
        pitch, yaw, roll = euler_angles
        if not (-90.0 <= pitch <= 90.0 and -90.0 <= yaw <= 90.0 and -45.0 <= roll <= 45.0):
            logger.warning("Pose out of physical constraints: Pitch=%.1f, Yaw=%.1f, Roll=%.1f", pitch, yaw, roll)
            return False
        return True

    @property
    def confidence(self) -> float:
        if not self._reprojection_errors:
            return 0.0
        avg_error = np.mean(self._reprojection_errors)
        return float(np.exp(-avg_error / 100.0))

    @staticmethod
    def draw_pose_point(frame: NDArray[np.uint8], result: HeadPoseResult, camera_matrix: NDArray[np.float32]) -> None:
        if result.confidence < 0.4 or result.landmarks is None:
            return
        nose_tip = tuple(result.landmarks[1].astype(int))
        cv2.circle(frame, nose_tip, 5, (0,0,255), -1)
        cv2.putText(frame, "Nose Tip", (nose_tip[0]+10, nose_tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    def draw_face_bounding_box(self, frame: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> None:
        x, y, w, h = cv2.boundingRect(landmarks.astype(np.int32))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Face", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    def process_frame(self, frame: NDArray[np.uint8]) -> Optional[HeadPoseResult]:
        logger.info("Processing frame of shape: %s", frame.shape)
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280,720))
            logger.info("Frame resized to 1280x720.")
        landmarks = self.detector.process_frame(frame)
        if not self._validate_landmarks(landmarks):
            logger.warning("Invalid landmarks; using fallback result.")
            return self._fallback_result(frame)
        if self._camera_matrix is None:
            self._initialize_camera_matrix(frame.shape[:2])
        image_points = landmarks[self._LANDMARK_INDICES]
        pnp_result = self._solve_pnp(image_points)
        if pnp_result is None:
            logger.warning("PnP solution not found; using fallback result.")
            return self._fallback_result(frame)
        rvec, tvec = pnp_result
        euler_angles = self._convert_to_euler(rvec)
        if not self._validate_physical_constraints(euler_angles):
            logger.warning("Pose out of physical constraints; using fallback result.")
            return self._fallback_result(frame)
        self._smooth_euler_angles(np.array(euler_angles, dtype=np.float32))
        if not self._is_pose_stable() and self._last_valid_result is not None:
            logger.info("Pose unstable; reverting to last valid result.")
            self._pose_history.clear()
            return self._last_valid_result
        result = HeadPoseResult(
            rvec=rvec,
            tvec=tvec,
            euler_angles=tuple(self._smoothed_euler.tolist()),
            landmarks=landmarks,
            confidence=1.0
        )
        self._last_valid_result = result
        if landmarks is not None:
            self.draw_face_bounding_box(frame, landmarks)
        if self._last_valid_result:
            self.draw_pose_point(frame, self._last_valid_result, self._camera_matrix)
        return self._last_valid_result

    def _fallback_result(self, frame: NDArray[np.uint8]) -> HeadPoseResult:
        if self._last_valid_result:
            logger.info("Using last valid head pose result as fallback.")
            return self._last_valid_result
        logger.info("No valid pose available; returning default result.")
        return HeadPoseResult(
            rvec=np.zeros((3,1), dtype=np.float32),
            tvec=np.zeros((3,1), dtype=np.float32),
            euler_angles=(0.0, 0.0, 0.0),
            landmarks=None,
            confidence=0.0
        )

    @staticmethod
    def draw_pose_axes(frame: NDArray[np.uint8], result: HeadPoseResult, camera_matrix: NDArray[np.float32], length: float = 0.1) -> NDArray[np.uint8]:
        if result.confidence < 0.4 or result.landmarks is None:
            return frame
        axis = np.float32([[length,0,0], [0,length,0], [0,0,length]])
        try:
            imgpts, _ = cv2.projectPoints(axis, result.rvec, result.tvec, camera_matrix, np.zeros((4,1), dtype=np.float32))
        except Exception as e:
            logger.error("Error projecting pose axes: %s", e, exc_info=True)
            return frame
        nose_point = tuple(result.landmarks[1].astype(int))
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        for i, color in enumerate(colors):
            end_point = tuple(imgpts[i].ravel().astype(int))
            cv2.line(frame, nose_point, end_point, color, 3)
        return frame

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with FaceMeshDetector() as detector:
        estimator = HeadPoseEstimator(detector)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error opening video stream.")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = estimator.process_frame(frame)
            if estimator._camera_matrix is not None and result and result.confidence > 0.4:
                frame = HeadPoseEstimator.draw_pose_axes(frame, result, estimator._camera_matrix)
            if result:
                pitch, yaw, roll = result.euler_angles
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                cv2.putText(frame, f"Confidence: {result.confidence:.2f}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.imshow("Head Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
