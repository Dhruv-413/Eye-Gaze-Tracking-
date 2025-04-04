# eye_blink_normalization.py
import cv2
import numpy as np
import logging
from collections import deque
from typing import Optional, Dict, Any, Union, List
from numpy.typing import NDArray

from core.face import FaceMeshDetector
from core.config import LEFT_EYE_IDX, RIGHT_EYE_IDX, EYE_BLINK_CONFIG
from utils.image_utils import draw_text
from utils.math_utils import calculate_ear, normalize_coordinates
from utils.logging_utils import configure_logging

logger = configure_logging("blink_detection.log")

class EyeBlinkDetector:
    """
    Detects eye blinks using the Eye Aspect Ratio (EAR) with temporal smoothing.
    
    This detector uses normalized landmarks (values in [0,1]) from the face detector.
    """
    def __init__(self, face_detector: FaceMeshDetector,
                 config: Optional[dict] = None):
        """
        Args:
            face_detector: Instance of FaceMeshDetector for landmark extraction.
            config: Configuration for blink detection; defaults to EYE_BLINK_CONFIG.
        """
        self.config = config if config is not None else EYE_BLINK_CONFIG
        self.face_detector = face_detector
        self.smoothing_window = self.config.SMOOTHING_WINDOW
        self.consec_frames = self.config.CONSEC_FRAMES
        self.closed_threshold = self.config.EAR_THRESHOLD
        self.ear_history = deque(maxlen=self.smoothing_window)
        self.frame_counter = 0
        self.blink_count = 0
        self.blink_in_progress = False
        self.ear_values = {'left': 0.0, 'right': 0.0, 'avg': 0.0, 'smoothed': 0.0}
        self.last_blink_time = 0
        # Normalization is always enabled, since we expect normalized landmarks.
        self.use_normalization = True

    def reset(self):
        """Reset the detector state."""
        self.ear_history.clear()
        self.frame_counter = 0
        self.blink_count = 0
        self.blink_in_progress = False
        self.ear_values = {'left': 0.0, 'right': 0.0, 'avg': 0.0, 'smoothed': 0.0}

    def process_frame(self, frame: NDArray[np.uint8], timestamp: float = None) -> Dict[str, Any]:
        """
        Process a frame to compute a smoothed normalized EAR and detect blinks.
        
        Returns:
            A dictionary with:
              - 'ear': Smoothed EAR.
              - 'ear_left': Left eye EAR.
              - 'ear_right': Right eye EAR.
              - 'blink_detected': Boolean flag indicating a blink.
              - 'blink_count': Total blink count.
              - 'landmarks': Dictionary with 'left_eye' and 'right_eye' normalized landmarks.
              - 'normalization_factor': Face width in pixels.
              - 'ear_asymmetry': Difference between left and right EAR.
        """
        result: Dict[str, Any] = {
            'ear': 0.0,
            'ear_left': 0.0,
            'ear_right': 0.0,
            'blink_detected': False,
            'blink_count': self.blink_count,
            'landmarks': None
        }
        
        # Get normalized landmarks from face detector.
        norm_landmarks = self.face_detector.process_frame(frame)
        if norm_landmarks is None:
            if self.frame_counter > 0:
                logger.debug("Face lost during potential blink; resetting counter.")
                self.frame_counter = 0
            return result

        try:
            h, w = frame.shape[:2]
            # Convert normalized landmarks to pixel coordinates for EAR calculation.
            left_eye = normalize_coordinates(norm_landmarks[LEFT_EYE_IDX], w, h)
            right_eye = normalize_coordinates(norm_landmarks[RIGHT_EYE_IDX], w, h)
            result['landmarks'] = {'left_eye': left_eye, 'right_eye': right_eye}
            
            # Compute face width from all x-coordinates.
            all_x = norm_landmarks[:, 0] * w
            face_width = max(all_x) - min(all_x)
            result['normalization_factor'] = face_width
            
            # Compute EAR for both eyes.
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            current_ear = (left_ear + right_ear) / 2.0
            
            self.ear_values['left'] = left_ear
            self.ear_values['right'] = right_ear
            self.ear_values['avg'] = current_ear
            result['ear_left'] = left_ear
            result['ear_right'] = right_ear

            # Temporal smoothing using median.
            self.ear_history.append(current_ear)
            smoothed_ear = np.median(self.ear_history) if len(self.ear_history) > 1 else current_ear
            self.ear_values['smoothed'] = smoothed_ear
            result['ear'] = smoothed_ear

            # Blink detection using state tracking.
            if not self.blink_in_progress and smoothed_ear < self.closed_threshold:
                self.frame_counter = 1
                self.blink_in_progress = True
                logger.debug(f"Potential blink started, EAR: {smoothed_ear:.3f}")
            elif self.blink_in_progress:
                if smoothed_ear < self.closed_threshold:
                    self.frame_counter += 1
                    logger.debug(f"Blink in progress, count: {self.frame_counter}, EAR: {smoothed_ear:.3f}")
                else:
                    if self.frame_counter >= self.consec_frames:
                        self.blink_count += 1
                        result['blink_detected'] = True
                        if timestamp is not None and self.last_blink_time > 0:
                            result['blink_interval'] = timestamp - self.last_blink_time
                        self.last_blink_time = timestamp if timestamp is not None else 0
                        logger.debug(f"Blink detected! Total count: {self.blink_count}")
                    else:
                        logger.debug(f"Blink not sustained: {self.frame_counter} frames")
                    self.frame_counter = 0
                    self.blink_in_progress = False

            result['blink_count'] = self.blink_count
            result['blink_in_progress'] = self.blink_in_progress
            result['ear_asymmetry'] = abs(left_ear - right_ear)
            return result
        
        except Exception as e:
            logger.error("Error during EAR calculation: %s", e, exc_info=True)
            return result

def run_eye_blink_detection(camera_id: int = 0, resolution: tuple = (1280, 720)) -> bool:
    """
    Run the eye blink detection demo using the webcam.
    
    Captures frames, processes each for blink detection using normalized EAR,
    overlays EAR and blink count, and displays the result.
    
    Args:
        camera_id (int): The camera device ID.
        resolution (tuple): Desired frame resolution (width, height).
        
    Returns:
        bool: True if the demo runs successfully, False otherwise.
    """
    logger.info(f"Starting eye blink detection demo on camera {camera_id} at resolution {resolution}.")
    logger.info("Using normalized landmarks exclusively.")
    
    with FaceMeshDetector() as detector:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Error: Could not open camera (device ID: {camera_id}).")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
            logger.warning(f"Could not determine FPS; assuming {fps} FPS.")
        
        blink_detector = EyeBlinkDetector(detector)
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame from camera.")
                    break
                frame = cv2.resize(frame, resolution)
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                blink_result = blink_detector.process_frame(frame, timestamp=current_time)
                frame_count += 1

                draw_text(frame, f"EAR: {blink_result.get('ear', 0):.2f}", (10, 30), (0, 255, 0))
                draw_text(frame, f"Blinks: {blink_result.get('blink_count', 0)}", (10, 70), (0, 255, 0))
                if 'blink_interval' in blink_result:
                    draw_text(frame, f"Interval: {blink_result['blink_interval']:.2f}s", (10, 110), (0, 255, 0))
                draw_text(frame, f"Asym: {blink_result.get('ear_asymmetry', 0):.2f}", (10, 150), (0, 255, 0))
                
                if blink_result.get('blink_in_progress', False):
                    draw_text(frame, "BLINK", (resolution[0] - 150, 70), (0, 0, 255))
                
                ear_bar_length = int(blink_result.get('ear', 0) * 100)
                cv2.rectangle(frame, (10, 180), (10 + ear_bar_length, 190), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 180), (110, 190), (255, 255, 255), 1)
                
                threshold_pos = int(blink_detector.closed_threshold * 100)
                cv2.line(frame, (10 + threshold_pos, 175), (10 + threshold_pos, 195), (0, 0, 255), 2)
                cv2.imshow("Eye Blink Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exit requested by user.")
                    break
                elif key == ord('r'):
                    logger.info("Resetting blink detector.")
                    blink_detector.reset()
                    start_time = current_time
            # End of loop.
        except Exception as e:
            logger.error("Error during processing: %s", e, exc_info=True)
            return False
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Resources released successfully.")
    
    return True

if __name__ == "__main__":
    run_eye_blink_detection()
