# eye_pupil_extract.py
import cv2
import numpy as np
from typing import NamedTuple, Optional, Tuple
from numpy.typing import NDArray
from core.face_tracker import FaceMeshDetector
from Interaction.constants import FaceLandmarks, DEFAULT  # Import DEFAULT explicitly

class EyeRegion(NamedTuple):
    image: NDArray[np.uint8]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: NDArray[np.float32]
    is_valid: bool

class EyeRegionExtractor:
    """
    Enhanced eye region extractor with error handling and adaptive processing
    
    Features:
    - Robust bounding box calculations with safety checks
    - Adaptive image preprocessing
    - Landmark validation
    - Configurable region padding
    - Detailed return structure
    """
    
    def __init__(
        self,
        face_detector: FaceMeshDetector,
        padding_ratio: float = 0.4,
        min_eye_size: int = 20,
        config: FaceLandmarks = DEFAULT  # Use DEFAULT directly
    ):
        """
        Args:
            face_detector: Initialized FaceMeshDetector instance
            padding_ratio: Percentage of region size to add as padding
            min_eye_size: Minimum acceptable eye region size in pixels
            config: Landmark configuration to use
        """
        self.face_detector = face_detector
        self.padding_ratio = padding_ratio
        self.min_eye_size = min_eye_size
        self.config = config
        
        # Precompute indices for performance
        self.left_eye_indices = config.LEFT_EYE
        self.right_eye_indices = config.RIGHT_EYE

    def _process_eye_region(
        self,
        frame: NDArray[np.uint8],
        landmarks: NDArray[np.float32],
        eye_indices: NDArray[np.int32]
    ) -> EyeRegion:
        """Process a single eye region with safety checks"""
        cropped_eye = self.face_detector.crop_eye_region(frame, landmarks, eye_indices)
        if cropped_eye is None or cropped_eye.size == 0:
            return EyeRegion(np.array([]), (0, 0, 0, 0), np.array([]), False)

        # Preprocess the cropped eye region
        processed = self._preprocess_region(cropped_eye)
        bbox = cv2.boundingRect(landmarks[eye_indices].astype(np.int32))
        return EyeRegion(processed, bbox, landmarks[eye_indices], True)

    def _safe_bounding_rect(
        self,
        points: NDArray[np.int32],
        frame_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Calculate bounding box with padding and safety checks"""
        x, y, w, h = cv2.boundingRect(points)
        
        # Add proportional padding
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)
        
        # Calculate safe coordinates
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_shape[1], x + w + pad_x)
        y2 = min(frame_shape[0], y + h + pad_y)
        
        return (x1, y1, x2 - x1, y2 - y1)

    def _preprocess_region(self, region: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply adaptive preprocessing to eye region"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Normalization
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized

    def extract_eye_regions(
        self,
        frame: NDArray[np.uint8]
    ) -> Tuple[EyeRegion, EyeRegion]:
        """
        Extract and preprocess both eye regions
        
        Returns:
            Tuple of (left_eye, right_eye) EyeRegion objects
        """
        landmarks = self.face_detector.process_frame(frame)
        if landmarks is None:
            return EyeRegion(), EyeRegion()

        left_eye = self._process_eye_region(frame, landmarks, self.left_eye_indices)
        right_eye = self._process_eye_region(frame, landmarks, self.right_eye_indices)
        
        return left_eye, right_eye

    @staticmethod
    def detect_pupil(eye_region: EyeRegion) -> Optional[Tuple[int, int]]:
        """Detect pupil center using adaptive thresholding"""
        if not eye_region.is_valid or eye_region.image.size == 0:
            return None

        # Thresholding
        _, thresh = cv2.threshold(eye_region.image, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        
        if moments["m00"] == 0:
            return None
            
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # Convert to original image coordinates
        return (eye_region.bbox[0] + cx, eye_region.bbox[1] + cy)

def main() -> None:
    """Example usage with visualization"""
    with FaceMeshDetector() as detector:
        extractor = EyeRegionExtractor(
            face_detector=detector,
            padding_ratio=0.3,
            min_eye_size=30
        )
        
        with cv2.VideoCapture(0) as cap:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                left_eye, right_eye = extractor.extract_eye_regions(frame)
                
                # Draw pupil detection
                for eye in [left_eye, right_eye]:
                    if eye.is_valid:
                        pupil = EyeRegionExtractor.detect_pupil(eye)
                        if pupil:
                            cv2.circle(frame, pupil, 3, (0,0,255), -1)
                
                # Display eye regions
                if left_eye.is_valid:
                    cv2.imshow("Left Eye", left_eye.image)
                if right_eye.is_valid:
                    cv2.imshow("Right Eye", right_eye.image)
                
                cv2.imshow("Pupil Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    main()