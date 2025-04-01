# eye_pupil_extract.py
import cv2
import numpy as np
import logging
from typing import NamedTuple, Optional, Tuple
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector
from Interaction.constants import FaceLandmarks, DEFAULT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EyeRegion(NamedTuple):
    image: NDArray[np.uint8]
    bbox: Tuple[int, int, int, int]
    landmarks: NDArray[np.float32]
    is_valid: bool

class EyeRegionExtractor:
    """
    Enhanced eye region extractor with Haar Cascade refinement and adaptive preprocessing.
    """
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        padding_ratio: float = 0.4,
        min_eye_size: int = 20,
        config: FaceLandmarks = DEFAULT
    ):
        self.face_detector = face_detector
        self.padding_ratio = padding_ratio
        self.min_eye_size = min_eye_size
        self.config = config
        self.left_eye_indices = config.LEFT_EYE
        self.right_eye_indices = config.RIGHT_EYE
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(cascade_path)
        if self.eye_cascade.empty():
            self.logger.error("Failed to load Haar Cascade from %s", cascade_path)

    def _process_eye_region(
        self,
        frame: NDArray[np.uint8],
        landmarks: NDArray[np.float32],
        eye_indices: NDArray[np.int32]
    ) -> EyeRegion:
        try:
            eye_points = landmarks[eye_indices]
            x, y, w, h = cv2.boundingRect(eye_points.astype(np.int32))
            pad = int(self.padding_ratio * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            if w < self.min_eye_size or h < self.min_eye_size or (x2-x1) <= 0 or (y2-y1) <= 0:
                self.logger.warning("Eye region too small or invalid.")
                return EyeRegion(np.array([]), (0,0,0,0), np.array([]), False)
            cropped_eye = frame[y1:y2, x1:x2]
            refined_bbox = self._refine_with_haar(cropped_eye)
            if refined_bbox is not None:
                ex, ey, ew, eh = refined_bbox
                x_refined = x1 + ex
                y_refined = y1 + ey
                cropped_eye = frame[y_refined:y_refined+eh, x_refined:x_refined+ew]
                x, y, w, h = x_refined, y_refined, ew, eh
                self.logger.debug("Eye region refined using Haar Cascade: %s", (x, y, w, h))
            processed = self._preprocess_region(cropped_eye)
            return EyeRegion(processed, (x, y, w, h), eye_points, True)
        except Exception as e:
            self.logger.error("Error processing eye region: %s", e, exc_info=True)
            return EyeRegion(np.array([]), (0,0,0,0), np.array([]), False)

    def _refine_with_haar(self, eye_region: NDArray[np.uint8]) -> Optional[Tuple[int,int,int,int]]:
        try:
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray_eye, scaleFactor=1.1, minNeighbors=3, minSize=(self.min_eye_size, self.min_eye_size))
            if len(eyes) == 0:
                self.logger.debug("No eyes detected by Haar Cascade.")
                return None
            best = max(eyes, key=lambda r: r[2] * r[3])
            self.logger.debug("Haar Cascade detection: %s", best)
            return best
        except Exception as e:
            self.logger.error("Error during Haar Cascade refinement: %s", e, exc_info=True)
            return None

    def _preprocess_region(self, region: NDArray[np.uint8]) -> NDArray[np.uint8]:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        normalized = cv2.normalize(morphed, None, 0, 255, cv2.NORM_MINMAX)
        return normalized

    @staticmethod
    def draw_eye_bounding_box(frame: NDArray[np.uint8], eye_region: EyeRegion) -> None:
        if not eye_region.is_valid:
            return
        x, y, w, h = eye_region.bbox
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, "Eye", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    @staticmethod
    def draw_pupil_bounding_box(frame: NDArray[np.uint8], pupil: Optional[Tuple[int, int, Tuple[int,int,int,int]]]) -> None:
        if pupil is None:
            return
        cx, cy, (x,y,w,h) = pupil
        # cv2.circle(frame, (cx,cy), 3, (0,255,0), -1)
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
        # cv2.putText(frame, "Pupil", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    def extract_eye_regions(self, frame: NDArray[np.uint8]) -> Tuple[EyeRegion, EyeRegion]:
        self.logger.info("Processing frame of shape: %s", frame.shape)
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, 720))
            self.logger.info("Frame resized to 1280x720.")
        landmarks = self.face_detector.process_frame(frame)
        if landmarks is None:
            self.logger.warning("No landmarks detected.")
            empty = EyeRegion(np.array([]), (0,0,0,0), np.array([]), False)
            return empty, empty
        left_eye = self._process_eye_region(frame, landmarks, self.left_eye_indices)
        right_eye = self._process_eye_region(frame, landmarks, self.right_eye_indices)
        self.draw_eye_bounding_box(frame, left_eye)
        self.draw_eye_bounding_box(frame, right_eye)
        self.logger.info("Extracted eye regions.")
        return left_eye, right_eye
