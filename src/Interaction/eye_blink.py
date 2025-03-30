# eye_blink.py
import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector
from core.utils import calculate_ear
from Interaction.constants import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX

class EyeBlinkDetector:
    """
    Robust eye blink detector using EAR (Eye Aspect Ratio) with adaptive thresholding and temporal smoothing.
    """
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        ear_threshold: Optional[float] = None,
        smoothing_window: int = 5,
        consec_frames: int = 3,
        calibration_frames: int = 30
    ):
        """
        Args:
            face_detector: Initialized FaceMeshDetector instance.
            ear_threshold: Optional fixed EAR threshold (auto-calibrates if None).
            smoothing_window: Number of frames for EAR moving average.
            consec_frames: Consecutive low-EAR frames needed to register a blink.
            calibration_frames: Frames to use for auto-calibration.
        """
        self.face_detector = face_detector
        self.consec_frames = consec_frames
        self.calibration_frames = calibration_frames

        # State management
        self.ear_history = deque(maxlen=smoothing_window)
        self.frame_counter = 0
        self.blink_count = 0
        self.calibration_values = []

        # Threshold handling
        self.ear_threshold = ear_threshold
        self._dynamic_threshold = None

    def _get_eye_landmarks(
        self,
        landmarks: NDArray[np.float32]
    ) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """Safely extract eye landmarks with bounds checking."""
        try:
            left_eye = landmarks[LEFT_EYE_EAR_IDX]
            right_eye = landmarks[RIGHT_EYE_EAR_IDX]
            return left_eye, right_eye
        except (IndexError, TypeError):
            return None, None

    def _calculate_smoothed_ear(self, left_eye: NDArray, right_eye: NDArray) -> float:
        """Calculate and smooth EAR using a moving average."""
        try:
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            current_ear = (left_ear + right_ear) / 2.0
            print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Current EAR: {current_ear:.2f}")
            self.ear_history.append(current_ear)
            return np.mean(self.ear_history)
        except (ValueError, TypeError) as e:
            print(f"Error calculating EAR: {e}")
            return 0.0

    def _update_calibration(self, ear: float) -> None:
        """Dynamic threshold calibration during initial frames."""
        if len(self.calibration_values) < self.calibration_frames:
            self.calibration_values.append(ear)
            if len(self.calibration_values) == self.calibration_frames:
                self._dynamic_threshold = np.mean(self.calibration_values) * 0.7
                print(f"Calibration complete. Dynamic EAR threshold set to {self._dynamic_threshold:.2f}")

    def _detect_blink(self, smoothed_ear: float) -> bool:
        """Core blink detection logic."""
        threshold = self.ear_threshold or self._dynamic_threshold

        if not threshold:
            return False

        if smoothed_ear < threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_count += 1
                self.frame_counter = 0  # Reset counter after detecting a blink
                return True
            self.frame_counter = 0  # Reset counter if EAR is above threshold

        return False

    def process_frame(self, frame: NDArray[np.uint8]) -> dict:
        """
        Process frame and return blink detection results with bounding boxes.
        """
        # Log frame dimensions for debugging
        print(f"Processing frame of shape: {frame.shape}")

        # Preprocess frame if needed (e.g., resizing or normalization)
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, 720))
            print("Frame resized to 1280x720 for consistent processing.")

        # Validate landmarks
        landmarks = self.face_detector.process_frame(frame)
        if landmarks is None:
            return {
                "frame": frame,
                "blink_detected": False,
                "ear": 0.0,
                "blink_count": self.blink_count,
                "threshold": self.ear_threshold or self._dynamic_threshold,
                "eye_boxes": []
            }

        left_eye, right_eye = self._get_eye_landmarks(landmarks)
        if left_eye is None or right_eye is None:
            return {
                "frame": frame,
                "blink_detected": False,
                "ear": 0.0,
                "blink_count": self.blink_count,
                "threshold": self.ear_threshold or self._dynamic_threshold,
                "eye_boxes": []
            }

        # Calculate EAR
        smoothed_ear = self._calculate_smoothed_ear(left_eye, right_eye)
        result = {
            "frame": frame.copy(),
            "blink_detected": False,
            "ear": smoothed_ear,
            "blink_count": self.blink_count,
            "threshold": self.ear_threshold or self._dynamic_threshold,
            "eye_boxes": []
        }

        # Update dynamic threshold during calibration
        if not self.ear_threshold and self._dynamic_threshold is None:
            self._update_calibration(smoothed_ear)
            return result

        # Detect blink
        result["blink_detected"] = self._detect_blink(smoothed_ear)

        # Draw bounding boxes for eyes
        for eye, label in zip([left_eye, right_eye], ["Left Eye", "Right Eye"]):
            x, y, w, h = cv2.boundingRect(eye.astype(np.int32))
            result["eye_boxes"].append((x, y, w, h))
            cv2.rectangle(result["frame"], (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result["frame"], label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return result

    @staticmethod
    def draw_annotations(
        frame: NDArray[np.uint8],
        ear: float,
        blink_count: int,
        threshold: float
    ) -> None:
        """Visualization layer for blink detection."""
        y_offset = 30
        cv2.putText(frame, f"Blinks: {blink_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f} (Thresh: {threshold:.2f})", (10, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

def main() -> None:
    """Example usage with proper resource management."""
    with FaceMeshDetector() as detector:
        blink_detector = EyeBlinkDetector(detector)
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = blink_detector.process_frame(frame)
            if result["ear"] > 0:  # Only draw if valid data
                EyeBlinkDetector.draw_annotations(
                    result["frame"],
                    result["ear"],
                    result["blink_count"],
                    result["threshold"]
                )

            cv2.imshow("Blink Detection", result["frame"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()