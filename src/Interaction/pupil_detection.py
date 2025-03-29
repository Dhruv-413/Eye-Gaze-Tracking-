# pupil_detection.py
import cv2
import numpy as np
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
    Enhanced pupil detection system with temporal smoothing and confidence scoring
    
    Features:
    - Adaptive thresholding with CLAHE preprocessing
    - Circularity-based contour validation
    - Temporal smoothing of pupil positions
    - Confidence scoring for detections
    - Configurable parameters
    - Resource management
    """
    
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        eye_extractor: EyeRegionExtractor,
        config: PupilConfig = PupilConfig(),
        landmarks_config=DEFAULT  # Use DEFAULT directly
    ):
        self.detector = face_detector
        self.eye_extractor = eye_extractor
        self.config = config
        self.landmarks_config = landmarks_config
        
        # State management
        self._position_history = deque(maxlen=config.temporal_smoothing)
        self._confidence_history = deque(maxlen=config.temporal_smoothing)

    def _preprocess_eye(self, eye_region: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Enhanced preprocessing pipeline"""
        # Convert to grayscale if needed
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        
        # CLAHE normalization
        clahe = cv2.createCLAHE()
        clipLimit=self.config.clahe_clip_limit,
        tileGridSize=(8, 8)
        enhanced = clahe.apply(gray)
        
        # Adaptive blurring
        blur_size = self.config.blur_kernel_size | 1  # Ensure odd number
        blurred = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0)
        
        return blurred

    def _detect_candidates(self, processed_eye: NDArray[np.uint8]) -> List[Tuple[NDArray, float]]:
        """Find potential pupil candidates with quality scores"""
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            processed_eye, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_thresh_block,
            self.config.adaptive_thresh_c
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config.min_pupil_area < area < self.config.max_pupil_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity >= self.config.min_circularity:
                    candidates.append((cnt, circularity))
        
        return candidates

    def _calculate_pupil_center(self, contour: NDArray) -> Optional[Tuple[int, int]]:
        """Calculate centroid with moment validation"""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def detect_pupil(self, eye_region: NDArray[np.uint8]) -> PupilDetectionResult:
        """Main detection pipeline"""
        processed = self._preprocess_eye(eye_region)
        candidates = self._detect_candidates(processed)
        
        best_confidence = 0.0
        best_contour = None
        best_center = None
        
        for cnt, circularity in candidates:
            center = self._calculate_pupil_center(cnt)
            if center:
                # Confidence based on circularity and area
                area = cv2.contourArea(cnt)
                area_conf = 1 - abs(area - 150) / 150  # Peak at 150px area
                confidence = (circularity + area_conf) / 2
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_contour = cnt
                    best_center = center
        
        return PupilDetectionResult(
            pupil_center=best_center,
            contour=best_contour,
            confidence=best_confidence,
            processed_eye=processed if self.config.show_processing else None
        )

    def process_frame(self, frame: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], List[PupilDetectionResult]]:
        """Process frame and return annotated results"""
        landmarks = self.detector.process_frame(frame)
        if landmarks is None:
            return frame, []
        
        left_eye, right_eye = self.eye_extractor.extract_eye_regions(frame)
        results = []
        
        for eye_region, pos in zip([left_eye, right_eye], ["left", "right"]):
            if not eye_region.is_valid:
                results.append(PupilDetectionResult(None, None, 0.0, None))
                continue
                
            result = self.detect_pupil(eye_region.image)
            results.append(result)
            
            # Temporal smoothing
            if result.pupil_center:
                self._position_history.append(result.pupil_center)
                self._confidence_history.append(result.confidence)
                
            # Draw results
            if result.pupil_center and result.confidence > 0.4:
                cx, cy = result.pupil_center
                cv2.circle(eye_region.image, (cx, cy), 3, (0, 255, 0), -1)
                cv2.drawContours(eye_region.image, [result.contour], -1, (255, 0, 0), 1)
        
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