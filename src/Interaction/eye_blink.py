# eye_blink.py
import cv2
import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple
from numpy.typing import NDArray

from core.face_tracker import FaceMeshDetector
from core.utils import calculate_ear
from Interaction.constants import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        calibration_frames: int = 30,
        draw_eye_bounding_box=True
    ):
        self.face_detector = face_detector
        self.consec_frames = consec_frames
        self.calibration_frames = calibration_frames
        self.ear_history = deque(maxlen=smoothing_window)
        self.frame_counter = 0
        self.blink_count = 0
        self.calibration_values = []
        self.ear_threshold = ear_threshold
        self._dynamic_threshold = None
        self.draw_eye_bounding_box = draw_eye_bounding_box

    def _get_eye_landmarks(self, landmarks: NDArray[np.float32]) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        try:
            left_eye = landmarks[LEFT_EYE_EAR_IDX]
            right_eye = landmarks[RIGHT_EYE_EAR_IDX]
            return left_eye, right_eye
        except Exception:
            return None, None

    def _calculate_smoothed_ear(self, left_eye: NDArray, right_eye: NDArray) -> float:
        try:
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            current_ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(current_ear)
            return np.mean(self.ear_history)
        except Exception as e:
            logger.error("Error calculating EAR: %s", e, exc_info=True)
            return 0.0

    def _update_calibration(self, ear: float) -> None:
        if len(self.calibration_values) < self.calibration_frames:
            self.calibration_values.append(ear)
            if len(self.calibration_values) == self.calibration_frames:
                self._dynamic_threshold = np.mean(self.calibration_values) * 0.6
                logger.info("Calibration complete. Dynamic EAR threshold set to %.2f", self._dynamic_threshold)

    def _detect_blink(self, smoothed_ear: float) -> bool:
        threshold = self.ear_threshold or self._dynamic_threshold
        if threshold is None:
            return False
        if smoothed_ear < threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_count += 1
                self.frame_counter = 0
                return True
            self.frame_counter = 0
        return False

    def process_frame(self, frame: NDArray[np.uint8]) -> dict:
        landmarks = self.face_detector.process_frame(frame)
        result = {
            "frame": frame.copy(),
            "blink_detected": False,
            "ear": 0.0,
            "blink_count": self.blink_count,
            "threshold": self.ear_threshold or self._dynamic_threshold,
            "eye_boxes": []
        }
        if landmarks is None:
            return result

        left_eye, right_eye = self._get_eye_landmarks(landmarks)
        if left_eye is None or right_eye is None:
            return result

        smoothed_ear = self._calculate_smoothed_ear(left_eye, right_eye)
        result["ear"] = smoothed_ear
        if self.ear_threshold is None and self._dynamic_threshold is None:
            self._update_calibration(smoothed_ear)
            return result

        result["blink_detected"] = self._detect_blink(smoothed_ear)

        for eye, label in zip([left_eye, right_eye], ["Left Eye", "Right Eye"]):
            try:
                x, y, w, h = cv2.boundingRect(eye.astype(np.int32))
                result["eye_boxes"].append((x, y, w, h))
                cv2.rectangle(result["frame"], (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(result["frame"], label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            except Exception as e:
                logger.error("Error drawing eye bounding box: %s", e, exc_info=True)
        return result
