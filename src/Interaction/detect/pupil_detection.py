import cv2
import numpy as np
from typing import NamedTuple, Optional, Tuple, List
from numpy.typing import NDArray
from core.config import PUPIL_CONFIG
from utils.logging_utils import configure_logging

logger = configure_logging("pupil_detection.log")

class PupilDetectionResult(NamedTuple):
    """
    Result of a pupil detection operation.
    """
    pupil_center: Optional[Tuple[int, int]]
    bounding_box: Optional[Tuple[int, int, int, int]]
    confidence: float
    radius: Optional[float] = None  # Average radius of the pupil.
    processed_eye: Optional[NDArray[np.uint8]] = None  # Preprocessed eye image.
    landmarks: Optional[NDArray[np.float32]] = None      # Contour points (optional).
    ellipse: Optional[Tuple] = None  # Fitted ellipse parameters (center, axes, angle)

class PupilDetector:
    """
    Detects the pupil within an eye image using adaptive thresholding,
    contour analysis, and optional ellipse fitting for a more accurate estimate.
    """
    def __init__(self,
                 config: Optional[dict] = None,
                 use_ellipse_fit: bool = True):
        """
        Args:
            config: Pupil detection configuration. Uses default if not provided.
            use_ellipse_fit: Whether to use ellipse fitting for improved pupil localization.
        """
        self.config = config if config is not None else PUPIL_CONFIG
        self.use_ellipse_fit = use_ellipse_fit
        self.prev_pupils: List[PupilDetectionResult] = []  # For temporal filtering

    def preprocess_eye_image(self, eye_image: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Preprocess the eye image to enhance pupil visibility.
        
        Returns:
            A tuple of (enhanced grayscale image, thresholded binary image).
        """
        # Convert to grayscale if necessary.
        if len(eye_image.shape) == 3:
            gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_image.copy()
        
        # Enhance contrast using CLAHE.
        clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur.
        blur_kernel = self.config.blur_kernel_size | 1  # Ensure kernel size is odd.
        blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)
        
        # Adaptive thresholding.
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_thresh_block,
            self.config.adaptive_thresh_c
        )
        
        # Morphological cleanup.
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return enhanced, thresh

    def filter_contours(self, contours: List) -> List[Tuple]:
        """
        Filter and score contours based on pupil-like properties.
        
        Returns:
            List of tuples: (contour, score, center, radius, ellipse)
        """
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_pupil_area or area > self.config.max_pupil_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.config.min_circularity:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            convexity = float(area) / hull_area if hull_area > 0 else 0
            if convexity < 0.8:
                continue
            
            # Determine pupil center and radius.
            if self.use_ellipse_fit and len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    axes = ellipse[1]
                    radius = (axes[0] + axes[1]) / 4.0
                    eccentricity = abs(axes[0] - axes[1]) / max(axes)
                    if eccentricity > 0.5:
                        continue
                except Exception:
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    distances = [cv2.norm(np.array(center) - pt[0]) for pt in cnt]
                    radius = float(np.mean(distances))
                    ellipse = None
            else:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                distances = [cv2.norm(np.array(center) - pt[0]) for pt in cnt]
                radius = float(np.mean(distances))
                ellipse = None
            
            # Score based on area, circularity, and a simple center preference (if applicable).
            ideal_area = (self.config.min_pupil_area + self.config.max_pupil_area) / 2.0
            size_score = 1.0 - min(abs(area - ideal_area) / ideal_area, 1.0)
            circularity_score = min(circularity, 1.0)
            convexity_score = convexity
            final_score = 0.3 * size_score + 0.3 * circularity_score + 0.2 * convexity_score
            
            candidates.append((cnt, final_score, center, radius, ellipse))
        
        # Sort candidates by score in descending order.
        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def apply_temporal_filtering(self, result: Optional[PupilDetectionResult]) -> Optional[PupilDetectionResult]:
        """
        Apply simple temporal filtering to smooth pupil detections over time.
        """
        if not result:
            if self.prev_pupils:
                prev = self.prev_pupils[-1]
                new_confidence = prev.confidence * 0.8
                result = PupilDetectionResult(
                    pupil_center=prev.pupil_center,
                    bounding_box=prev.bounding_box,
                    confidence=new_confidence,
                    radius=prev.radius,
                    processed_eye=None,
                    landmarks=None,
                    ellipse=prev.ellipse
                )
            return result
        
        # Exponential smoothing if previous detections exist.
        if self.prev_pupils:
            alpha = 0.3
            prev = self.prev_pupils[-1]
            if prev.pupil_center and result.pupil_center:
                new_x = int(alpha * result.pupil_center[0] + (1 - alpha) * prev.pupil_center[0])
                new_y = int(alpha * result.pupil_center[1] + (1 - alpha) * prev.pupil_center[1])
                new_radius = alpha * result.radius + (1 - alpha) * prev.radius if result.radius and prev.radius else result.radius
                result = PupilDetectionResult(
                    pupil_center=(new_x, new_y),
                    bounding_box=result.bounding_box,
                    confidence=result.confidence,
                    radius=new_radius,
                    processed_eye=result.processed_eye,
                    landmarks=result.landmarks,
                    ellipse=result.ellipse
                )
                logger.debug("Applied temporal smoothing to pupil detection.")
        
        self.prev_pupils.append(result)
        if len(self.prev_pupils) > 5:
            self.prev_pupils.pop(0)
        return result

    def detect_pupil(self, eye_image: NDArray[np.uint8]) -> Optional[PupilDetectionResult]:
        """
        Detect the pupil from a given eye image.
        
        Args:
            eye_image: A cropped image of the eye (BGR or grayscale).
            
        Returns:
            A PupilDetectionResult with pupil center, bounding box, radius, and confidence,
            or None if no valid pupil is detected.
        """
        try:
            self.last_image_shape = eye_image.shape
            # Preprocess the eye image
            enhanced, thresh = self.preprocess_eye_image(eye_image)
            self.prev_image = enhanced  # Optionally store for debugging
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = self.filter_contours(contours)
            
            if not candidates:
                logger.debug("No valid pupil contour found.")
                return self.apply_temporal_filtering(None)
            
            best_contour, best_confidence, best_center, best_radius, best_ellipse = candidates[0]
            x, y, w, h = cv2.boundingRect(best_contour)
            if best_confidence < self.config.confidence_threshold:
                logger.debug(f"Pupil confidence {best_confidence:.2f} below threshold {self.config.confidence_threshold}.")
                return self.apply_temporal_filtering(None)
            
            result = PupilDetectionResult(
                pupil_center=best_center,
                bounding_box=(x, y, w, h),
                confidence=best_confidence,
                radius=best_radius,
                processed_eye=thresh,
                landmarks=best_contour,
                ellipse=best_ellipse
            )
            
            filtered_result = self.apply_temporal_filtering(result)
            if filtered_result:
                logger.info(f"Pupil detected at {filtered_result.pupil_center} with confidence {filtered_result.confidence:.2f}.")
            return filtered_result
        except Exception as e:
            logger.error(f"Error during pupil detection: {e}", exc_info=True)
            return None

    def draw_pupil_detection(self, frame: NDArray[np.uint8], result: PupilDetectionResult,
                              color: Tuple[int, int, int] = (0, 255, 0), 
                              show_details: bool = True) -> None:
        """
        Draw the pupil detection result on the frame.
        
        Args:
            frame: The image on which to draw.
            result: The pupil detection result.
            color: Color for drawn elements.
            show_details: Whether to display additional details (contour points, confidence, etc.)
        """
        if result is None:
            return
        
        x, y, w, h = result.bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        cx, cy = result.pupil_center
        cv2.circle(frame, (cx, cy), 2, color, -1)
        if result.radius:
            cv2.circle(frame, (cx, cy), int(result.radius), color, 1)
        if result.ellipse:
            cv2.ellipse(frame, result.ellipse, (0, 255, 255), 1)
        if show_details and result.landmarks is not None:
            for pt in result.landmarks:
                cv2.circle(frame, (pt[0][0], pt[0][1]), 1, (255, 0, 0), -1)
        cv2.putText(frame, f"Pupil: {result.confidence:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def run_pupil_detection(camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)) -> bool:
    """
    Run the pupil detection demo using the webcam.
    
    For demonstration, this function processes the entire frame as an eye image.
    In a full system, you would use a cropped eye image from a face detection module.
    
    Args:
        camera_id (int): The camera device ID.
        resolution (Tuple[int, int]): Desired frame resolution.
        
    Returns:
        bool: True if the demo runs successfully, False otherwise.
    """
    logger.info(f"Starting pupil detection demo using camera {camera_id} with resolution {resolution}.")
    pupil_detector = PupilDetector(use_ellipse_fit=True)
    pupil_detector.prev_pupils = []  # Initialize detection history
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Error: Could not open camera (device ID: {camera_id}).")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cv2.namedWindow("Pupil Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame from camera.")
                break
            frame = cv2.resize(frame, resolution)
            
            # For demo purposes, process the full frame as the eye image.
            result = pupil_detector.detect_pupil(frame)
            if result:
                pupil_detector.draw_pupil_detection(frame, result, color=(0, 255, 0))
                if result.processed_eye is not None:
                    processed_display = cv2.resize(result.processed_eye, (resolution[0] // 3, resolution[1] // 3))
                    cv2.imshow("Processed", processed_display)
            
            cv2.imshow("Pupil Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit requested.")
                break
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released.")
    
    return True

if __name__ == "__main__":
    run_pupil_detection()
