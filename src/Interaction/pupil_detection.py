# pupil_detection.py
import cv2
import numpy as np
import logging
from typing import NamedTuple, Optional, Tuple, List
from numpy.typing import NDArray
from collections import deque
from core.face_tracker import FaceMeshDetector
from Interaction.eye_pupil_extract import EyeRegionExtractor, EyeRegion
from Interaction.constants import FaceLandmarks, DEFAULT  # Import DEFAULT explicitly

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

        # State management
        self._position_history = deque(maxlen=config.temporal_smoothing)
        self._confidence_history = deque(maxlen=config.temporal_smoothing)

        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _preprocess_eye(self, eye_region: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], int]:
        """Preprocess the eye region with dynamic thresholding."""
        # Convert to grayscale if needed
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region

        # CLAHE normalization
        clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive blurring
        blur_size = self.config.blur_kernel_size | 1  # Ensure odd number
        blurred = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0)

        # Dynamically adjust thresholding parameter
        mean_intensity = np.mean(blurred)
        adjusted_c = self.config.adaptive_thresh_c + int((mean_intensity - 128) / 10)
        adjusted_c = max(1, min(adjusted_c, 10))  # Clamp to a reasonable range

        self.logger.debug(f"Preprocessed eye region with adjusted_c={adjusted_c}")
        return blurred, adjusted_c

    def _detect_candidates(self, processed_eye: NDArray[np.uint8], adjusted_c: int) -> Optional[Tuple[NDArray, float]]:
        """Find the best pupil candidate based on quality scores."""
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            processed_eye, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_thresh_block,
            adjusted_c
        )

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Debug: Display thresholded image
        if self.config.show_processing:
            cv2.imshow("Thresholded Eye", thresh)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_confidence = 0.0
        center_x, center_y = processed_eye.shape[1] // 2, processed_eye.shape[0] // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config.min_pupil_area < area < self.config.max_pupil_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                solidity = cv2.contourArea(cnt) / cv2.contourArea(cv2.convexHull(cnt))  # Solidity metric
                aspect_ratio = max(cv2.boundingRect(cnt)[2:]) / min(cv2.boundingRect(cnt)[2:])  # Aspect ratio

                if circularity >= self.config.min_circularity and solidity > 0.8 and aspect_ratio < 2.0:
                    # Calculate distance to center
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    distance_to_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

                    # Confidence based on circularity, area, and distance to center
                    area_conf = 1 - abs(area - 150) / 150  # Peak confidence at 150px area
                    distance_conf = 1 - (distance_to_center / max(center_x, center_y))
                    confidence = (circularity + area_conf + distance_conf) / 3

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_contour = cnt

        if best_contour is None:
            self.logger.warning("No valid pupil contour found.")
        else:
            self.logger.info(f"Selected contour with confidence: {best_confidence:.2f}")

        return (best_contour, best_confidence) if best_contour is not None else None

    def _calculate_pupil_center(self, contour: NDArray, eye_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Calculate centroid with moment validation and map to original frame coordinates."""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Map to original frame coordinates
        x_offset, y_offset, _, _ = eye_bbox
        return cx + x_offset, cy + y_offset

    def detect_pupil(self, eye_region: EyeRegion) -> PupilDetectionResult:
        """Main detection pipeline."""
        processed, adjusted_c = self._preprocess_eye(eye_region.image)
        candidate = self._detect_candidates(processed, adjusted_c)

        if candidate is None:
            return PupilDetectionResult(
                pupil_center=None,
                contour=None,
                confidence=0.0,
                processed_eye=processed if self.config.show_processing else None
            )

        best_contour, best_confidence = candidate
        pupil_center = self._calculate_pupil_center(best_contour, eye_region.bbox)

        self.logger.info(f"Pupil detected at {pupil_center} with confidence {best_confidence:.2f}")
        return PupilDetectionResult(
            pupil_center=pupil_center,
            contour=best_contour,
            confidence=best_confidence,
            processed_eye=processed if self.config.show_processing else None
        )

    def process_frame(self, frame: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], List[PupilDetectionResult]]:
        """Process frame and return annotated results."""
        landmarks = self.detector.process_frame(frame)
        if landmarks is None:
            return frame, []

        left_eye, right_eye = self.eye_extractor.extract_eye_regions(frame)
        results = []

        for eye_region, pos in zip([left_eye, right_eye], ["left", "right"]):
            if not eye_region.is_valid:
                results.append(PupilDetectionResult(None, None, 0.0, None))
                continue

            result = self.detect_pupil(eye_region)
            results.append(result)

            # Draw pupil bounding box
            if result.pupil_center and result.confidence > 0.4:
                EyeRegionExtractor.draw_pupil_bounding_box(frame, (result.pupil_center[0], result.pupil_center[1], cv2.boundingRect(result.contour)))

            # Debug: Log pupil detection results
            self.logger.info(f"{pos.capitalize()} Eye - Pupil Center: {result.pupil_center}, Confidence: {result.confidence:.2f}")

        self.logger.info(f"Processed frame with {len(results)} pupil detections.")
        return frame, results

    @property
    def smoothed_position(self) -> Optional[Tuple[float, float]]:
        """Get temporally smoothed pupil position"""
        if not self._position_history:
            return None
        return tuple(np.mean(self._position_history, axis=0))

    @property
    def smoothed_confidence(self) -> float:
        """Get smoothed confidence score"""
        if not self._confidence_history:
            return 0.0
        return np.mean(self._confidence_history)

def main() -> None:
    """Example usage with resource management"""
    config = PupilConfig(
        min_pupil_area=30,
        max_pupil_area=300,
        show_processing=True
    )
    
    with FaceMeshDetector() as detector:
        eye_extractor = EyeRegionExtractor(detector)
        pupil_detector = PupilDetector(detector, eye_extractor, config)
        
        with cv2.VideoCapture(0) as cap:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, results = pupil_detector.process_frame(frame)
                
                # Display processing
                for i, result in enumerate(results):
                    if result.processed_eye is not None:
                        cv2.imshow(f"Eye Processing {i}", result.processed_eye)
                cv2.imshow("Pupil Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    main()