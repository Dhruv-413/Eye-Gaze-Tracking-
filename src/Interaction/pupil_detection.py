# pupil_detection.py
import cv2
import numpy as np
import logging
from typing import NamedTuple, Optional, Tuple, List
from numpy.typing import NDArray
from collections import deque
from core.face_tracker import FaceMeshDetector
from Interaction.eye_pupil_extract import EyeRegionExtractor, EyeRegion
from Interaction.constants import FaceLandmarks, DEFAULT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PupilDetectionResult(NamedTuple):
    pupil_center: Optional[Tuple[int, int]]
    contour: Optional[NDArray[np.int32]]
    confidence: float
    processed_eye: Optional[NDArray[np.uint8]]

class PupilConfig(NamedTuple):
    min_pupil_area: int = 20
    max_pupil_area: int = 400
    min_circularity: float = 0.7
    adaptive_thresh_block: int = 11
    adaptive_thresh_c: int = 2
    clahe_clip_limit: float = 2.0
    blur_kernel_size: int = 5
    temporal_smoothing: int = 5
    show_processing: bool = False

class PupilDetector:
    """
    Enhanced pupil detection system with temporal smoothing and confidence scoring.
    """
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        eye_extractor: EyeRegionExtractor,
        config: PupilConfig = PupilConfig(),
        landmarks_config=DEFAULT
    ):
        self.detector = face_detector
        self.eye_extractor = eye_extractor
        self.config = config
        self.landmarks_config = landmarks_config
        self._position_history = deque(maxlen=config.temporal_smoothing)
        self._confidence_history = deque(maxlen=config.temporal_smoothing)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _preprocess_eye(self, eye_region: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], int]:
        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_region
        clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip_limit, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blur_size = self.config.blur_kernel_size | 1
        blurred = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0)
        mean_intensity = np.mean(blurred)
        adjusted_c = self.config.adaptive_thresh_c + int((mean_intensity - 128)/10)
        adjusted_c = max(1, min(adjusted_c, 10))
        logger.debug(f"Preprocessed eye: mean_intensity={mean_intensity:.2f}, adjusted_c={adjusted_c}")
        return blurred, adjusted_c

    def _detect_candidates(self, processed_eye: NDArray[np.uint8], adjusted_c: int) -> Optional[Tuple[NDArray, float]]:
        thresh = cv2.adaptiveThreshold(
            processed_eye, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_thresh_block,
            adjusted_c
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        if self.config.show_processing:
            cv2.imshow("Thresholded Eye", thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_confidence = 0.0
        center_x, center_y = processed_eye.shape[1]//2, processed_eye.shape[0]//2
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.config.min_pupil_area < area < self.config.max_pupil_area):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter**2)
            convex_hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area/convex_hull_area if convex_hull_area>0 else 0
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect_ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-5)
            if circularity >= self.config.min_circularity and solidity > 0.8 and aspect_ratio < 2.0:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                distance_to_center = np.sqrt((cx-center_x)**2 + (cy-center_y)**2)
                area_conf = 1 - abs(area - 150)/150
                distance_conf = 1 - (distance_to_center / max(center_x, center_y))
                confidence = (circularity + area_conf + distance_conf)/3
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_contour = cnt
        if best_contour is None:
            logger.warning("No valid pupil contour found.")
        else:
            logger.info("Selected contour with confidence: %.2f", best_confidence)
        return (best_contour, best_confidence) if best_contour is not None else None

    def _calculate_pupil_center(self, contour: NDArray, eye_bbox: Tuple[int,int,int,int]) -> Optional[Tuple[int,int]]:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        x_offset, y_offset, _, _ = eye_bbox
        return cx + x_offset, cy + y_offset

    def detect_pupil(self, eye_region: EyeRegion) -> PupilDetectionResult:
        processed, adjusted_c = self._preprocess_eye(eye_region.image)
        candidate = self._detect_candidates(processed, adjusted_c)
        if candidate is None:
            return PupilDetectionResult(None, None, 0.0, processed if self.config.show_processing else None)
        best_contour, best_confidence = candidate
        pupil_center = self._calculate_pupil_center(best_contour, eye_region.bbox)
        logger.info("Pupil detected at %s with confidence %.2f", pupil_center, best_confidence)
        if pupil_center:
            self._position_history.append(pupil_center)
        self._confidence_history.append(best_confidence)
        return PupilDetectionResult(pupil_center, best_contour, best_confidence, processed if self.config.show_processing else None)

    def process_frame(self, frame: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], List[PupilDetectionResult]]:
        landmarks = self.detector.process_frame(frame)
        if landmarks is None:
            return frame, []
        left_eye, right_eye = self.eye_extractor.extract_eye_regions(frame)
        results: List[PupilDetectionResult] = []
        for eye_region, pos in zip([left_eye, right_eye], ["left", "right"]):
            if not eye_region.is_valid:
                results.append(PupilDetectionResult(None, None, 0.0, None))
                continue
            result = self.detect_pupil(eye_region)
            results.append(result)
            if result.pupil_center and result.confidence > 0.4 and result.contour is not None:
                bbox = cv2.boundingRect(result.contour)
                EyeRegionExtractor.draw_pupil_bounding_box(frame, (result.pupil_center[0], result.pupil_center[1], bbox))
            logger.info("%s Eye - Pupil Center: %s, Confidence: %.2f", pos.capitalize(), result.pupil_center, result.confidence)
        logger.info("Processed frame with %d pupil detections.", len(results))
        return frame, results

    @property
    def smoothed_position(self) -> Optional[Tuple[float, float]]:
        if not self._position_history:
            return None
        return tuple(np.mean(self._position_history, axis=0))

    @property
    def smoothed_confidence(self) -> float:
        if not self._confidence_history:
            return 0.0
        return float(np.mean(self._confidence_history))

def main() -> None:
    import cv2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    config = PupilConfig(min_pupil_area=30, max_pupil_area=300, show_processing=True)
    from core.face_tracker import FaceMeshDetector
    face_detector = FaceMeshDetector()
    eye_extractor = EyeRegionExtractor(face_detector)
    pupil_detector = PupilDetector(face_detector, eye_extractor, config)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pupil_detector.logger.error("Error opening video stream.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, results = pupil_detector.process_frame(frame)
        for i, result in enumerate(results):
            if result.processed_eye is not None:
                cv2.imshow(f"Eye Processing {i}", result.processed_eye)
        cv2.imshow("Pupil Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
