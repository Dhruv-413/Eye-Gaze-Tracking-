# eye_pupil_extract.py
import cv2
import numpy as np
import logging
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
    Enhanced eye region extractor with error handling and adaptive processing.
    """

    def __init__(
        self,
        face_detector: FaceMeshDetector,
        padding_ratio: float = 0.4,
        min_eye_size: int = 20,
        config: FaceLandmarks = DEFAULT
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

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _process_eye_region(
        self,
        frame: NDArray[np.uint8],
        landmarks: NDArray[np.float32],
        eye_indices: NDArray[np.int32]
    ) -> EyeRegion:
        """Process a single eye region with safety checks."""
        try:
            eye_points = landmarks[eye_indices]
            x, y, w, h = self._safe_bounding_rect(eye_points.astype(np.int32), frame.shape[:2])

            # Validate minimum size and aspect ratio
            if w < self.min_eye_size or h < self.min_eye_size:
                self.logger.warning(f"Eye region too small: width={w}, height={h}. Skipping.")
                return EyeRegion(np.array([]), (0, 0, 0, 0), np.array([]), False)
            if w / h < 1.2:  # Relaxed aspect ratio check
                self.logger.warning(f"Eye region has invalid aspect ratio: width={w}, height={h}. Skipping.")
                return EyeRegion(np.array([]), (0, 0, 0, 0), np.array([]), False)

            cropped_eye = frame[y:y+h, x:x+w]
            processed = self._preprocess_region(cropped_eye)
            self.logger.debug(f"Processed eye region with bbox={x, y, w, h}")
            return EyeRegion(processed, (x, y, w, h), eye_points, True)
        except Exception as e:
            self.logger.error(f"Error processing eye region: {e}")
            return EyeRegion(np.array([]), (0, 0, 0, 0), np.array([]), False)

    def _safe_bounding_rect(
        self,
        points: NDArray[np.int32],
        frame_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Calculate bounding box with increased padding and safety checks."""
        x, y, w, h = cv2.boundingRect(points)
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)

        # Ensure padding is sufficient for small regions
        pad_x = max(pad_x, 10)
        pad_y = max(pad_y, 10)

        # Adjust bounding box dimensions dynamically if too small
        if w < self.min_eye_size or h < self.min_eye_size:
            self.logger.warning(f"Invalid bounding box: width={w}, height={h}. Adjusting padding.")
            pad_x = max(pad_x, self.min_eye_size - w)
            pad_y = max(pad_y, self.min_eye_size - h)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_shape[1], x + w + pad_x)
        y2 = min(frame_shape[0], y + h + pad_y)

        # Ensure the bounding box is valid
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            self.logger.warning("Adjusted bounding box is invalid. Skipping.")
            return 0, 0, 0, 0

        return x1, y1, x2 - x1, y2 - y1

    def _preprocess_region(self, region: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply adaptive preprocessing to eye region with reflection handling."""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        # Morphological operations to reduce glare and reflections
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        # Normalization
        normalized = cv2.normalize(morphed, None, 0, 255, cv2.NORM_MINMAX)

        return normalized

    @staticmethod
    def draw_eye_bounding_box(frame: NDArray[np.uint8], eye_region: EyeRegion) -> None:
        """Draw the bounding box for the eye region."""
        if not eye_region.is_valid:
            return
        x, y, w, h = eye_region.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for the eye
        cv2.putText(frame, "Eye", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    @staticmethod
    def draw_pupil_bounding_box(frame: NDArray[np.uint8], pupil: Optional[Tuple[int, int, Tuple[int, int, int, int]]]) -> None:
        """Draw the bounding box for the pupil."""
        if pupil is None:
            return
        cx, cy, (x, y, w, h) = pupil
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)  # Green dot for the pupil center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green box for the pupil
        cv2.putText(frame, "Pupil", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def extract_eye_regions(
        self,
        frame: NDArray[np.uint8]
    ) -> Tuple[EyeRegion, EyeRegion]:
        """
        Extract and preprocess both eye regions.
        """
        # Log frame dimensions for debugging
        self.logger.info(f"Processing frame of shape: {frame.shape}")

        # Preprocess frame if needed (e.g., resizing or normalization)
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, 720))
            self.logger.info("Frame resized to 1280x720 for consistent processing.")

        # Validate landmarks
        landmarks = self.face_detector.process_frame(frame)
        if landmarks is None:
            self.logger.warning("No landmarks detected.")
            return EyeRegion(np.array([]), (0, 0, 0, 0), np.array([]), False), \
                   EyeRegion(np.array([]), (0, 0, 0, 0), np.array([]), False)

        left_eye = self._process_eye_region(frame, landmarks, self.left_eye_indices)
        right_eye = self._process_eye_region(frame, landmarks, self.right_eye_indices)

        # Draw eye bounding boxes
        self.draw_eye_bounding_box(frame, left_eye)
        self.draw_eye_bounding_box(frame, right_eye)

        self.logger.info("Extracted eye regions.")
        return left_eye, right_eye

    @staticmethod
    def detect_pupil(eye_region: EyeRegion) -> Optional[Tuple[int, int, Tuple[int, int, int, int]]]:
        """Detect pupil center and return bounding box."""
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

        # Get bounding box for the pupil
        x, y, w, h = cv2.boundingRect(largest)
        return (eye_region.bbox[0] + cx, eye_region.bbox[1] + cy, (x, y, w, h))

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
                            EyeRegionExtractor.draw_pupil_bounding_box(frame, pupil)
                
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