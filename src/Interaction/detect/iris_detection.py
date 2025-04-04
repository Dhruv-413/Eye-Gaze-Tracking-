import cv2
import numpy as np
import time
import os
from typing import NamedTuple, Optional, Tuple, Dict, List
from numpy.typing import NDArray

from core.config import DEFAULT, IRIS_CONFIG
from utils.logging_utils import configure_logging

logger = configure_logging("iris_detection.log")

class IrisDetectionResult(NamedTuple):
    iris_center: Optional[Tuple[int, int]]
    bounding_box: Optional[Tuple[int, int, int, int]]
    confidence: float
    radius: Optional[float] = None  # Added radius field for better visualization
    landmarks: Optional[NDArray[np.float32]] = None  # Store the iris landmarks

class IrisDetector:
    """
    Detects iris regions using the iris landmarks provided by MediaPipe FaceMesh.
    It extracts the iris region based on the landmark indices in the configuration,
    computes a bounding box, center, and radius, and outputs a confidence measure.
    """
    def __init__(self,
                 min_area: Optional[float] = None,
                 max_area: Optional[float] = None,
                 confidence_threshold: Optional[float] = None,
                 use_ellipse_fit: bool = True):
        # Use configuration defaults if not provided
        self.min_area = min_area if min_area is not None else IRIS_CONFIG.min_iris_area
        self.max_area = max_area if max_area is not None else IRIS_CONFIG.max_iris_area
        self.confidence_threshold = (confidence_threshold if confidence_threshold is not None
                                     else IRIS_CONFIG.confidence_threshold)
        self.use_ellipse_fit = use_ellipse_fit
        
        # Detection history for tracking and smoothing
        self.detection_history: Dict[str, List[IrisDetectionResult]] = {
            "left": [],
            "right": []
        }
        self.history_size = 5  # Number of frames to keep in history
        
        logger.info(f"IrisDetector initialized with min_area={self.min_area}, "
                   f"max_area={self.max_area}, confidence_threshold={self.confidence_threshold}, "
                   f"use_ellipse_fit={self.use_ellipse_fit}")

    def detect_iris(self, landmarks: NDArray[np.float32], side: str = "left") -> Optional[IrisDetectionResult]:
        """
        Detect the iris for a given side ("left" or "right") using the iris landmarks.
        
        Args:
            landmarks: Facial landmarks array (pixel coordinates).
            side (str): "left" or "right" specifying which iris to detect.
        
        Returns:
            An IrisDetectionResult with the iris center, bounding box, radius, and confidence,
            or None if detection is unsuccessful.
        """
        try:
            if side.lower() == "left":
                iris_indices = DEFAULT.LEFT_IRIS
            elif side.lower() == "right":
                iris_indices = DEFAULT.RIGHT_IRIS
            else:
                logger.error(f"Invalid side '{side}' provided for iris detection. Use 'left' or 'right'.")
                return None

            if iris_indices.size == 0 or np.max(iris_indices) >= landmarks.shape[0]:
                logger.warning(f"Iris indices are invalid for the provided landmarks. "
                              f"Size: {iris_indices.size}, Max index: {np.max(iris_indices) if iris_indices.size > 0 else 'N/A'}, "
                              f"Landmarks shape: {landmarks.shape}")
                return None

            # Extract iris landmarks
            iris_landmarks = landmarks[iris_indices]
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(iris_landmarks.astype(np.int32))
            area = w * h
            
            # Check if the area is within acceptable range
            if area < self.min_area or area > self.max_area:
                logger.debug(f"{side.capitalize()} iris area {area:.2f} out of acceptable range "
                            f"({self.min_area}, {self.max_area}).")
                return None

            # Calculate the center and radius
            if self.use_ellipse_fit and len(iris_landmarks) >= 5:
                # Use ellipse fitting for more accurate iris center and shape
                ellipse = cv2.fitEllipse(iris_landmarks.astype(np.int32))
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                # Use average of major and minor axes for radius
                radius = (ellipse[1][0] + ellipse[1][1]) / 4  # Divide by 4 as ellipse params are diameters
            else:
                # Simple center calculation using mean of landmarks
                center_x = int(np.mean(iris_landmarks[:, 0]))
                center_y = int(np.mean(iris_landmarks[:, 1]))
                center = (center_x, center_y)
                
                # Calculate radius as average distance from center to all points
                distances = np.sqrt(np.sum((iris_landmarks - np.array(center)) ** 2, axis=1))
                radius = float(np.mean(distances))

            # Compute confidence measure based on area proximity
            mid_area = (self.min_area + self.max_area) / 2.0
            confidence = 1 - abs(area - mid_area) / mid_area
            
            # Apply additional confidence factors
            # 1. Circularity factor: ratio of width to height should be close to 1 for a circular iris
            circularity = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            # 2. Uniformity factor: std dev of distances from center should be low
            uniformity = 1 - min(1, np.std(distances) / (radius + 1e-5)) if radius > 0 else 0
            
            # Combine confidence factors
            confidence = 0.5 * confidence + 0.25 * circularity + 0.25 * uniformity
            
            if confidence < self.confidence_threshold:
                logger.debug(f"{side.capitalize()} iris confidence {confidence:.2f} is below threshold "
                            f"{self.confidence_threshold}.")
                return None

            result = IrisDetectionResult(
                iris_center=center,
                bounding_box=(x, y, w, h),
                confidence=confidence,
                radius=radius,
                landmarks=iris_landmarks
            )
            
            # Add to detection history for smoothing
            self.detection_history[side].append(result)
            if len(self.detection_history[side]) > self.history_size:
                self.detection_history[side].pop(0)
                
            logger.debug(f"{side.capitalize()} iris detected at {result.iris_center} "
                        f"with radius {radius:.2f}px and confidence {confidence:.2f}.")
            return result
            
        except Exception as e:
            logger.error(f"Error during {side} iris detection: {e}", exc_info=True)
            return None

    def get_smoothed_result(self, side: str) -> Optional[IrisDetectionResult]:
        """
        Get a smoothed iris detection result based on recent history.
        
        Args:
            side: 'left' or 'right' eye
            
        Returns:
            Smoothed IrisDetectionResult or None if no history
        """
        history = self.detection_history.get(side, [])
        if not history:
            return None
            
        # Weight recent detections more heavily
        weights = np.linspace(0.5, 1.0, len(history))
        weights = weights / weights.sum()
        
        # Calculate weighted average of centers and radii
        centers = np.array([r.iris_center for r in history])
        center_x = int(np.sum(centers[:, 0] * weights))
        center_y = int(np.sum(centers[:, 1] * weights))
        
        radii = np.array([r.radius for r in history])
        avg_radius = float(np.sum(radii * weights))
        
        # Use most recent bounding box and confidence
        recent = history[-1]
        
        return IrisDetectionResult(
            iris_center=(center_x, center_y),
            bounding_box=recent.bounding_box,
            confidence=recent.confidence,
            radius=avg_radius,
            landmarks=recent.landmarks
        )

    def draw_iris_detection(self, frame: NDArray[np.uint8], result: IrisDetectionResult,
                           color: Tuple[int, int, int] = (255, 0, 0),
                           show_landmarks: bool = True,
                           label: str = "") -> None:
        """
        Draw the iris detection result on the frame.
        
        Args:
            frame: The image on which to draw
            result: The iris detection result
            color: Color of the drawn elements (BGR)
            show_landmarks: Whether to draw iris landmark points
            label: Optional label to add ("Left" or "Right")
        """
        if result is None:
            return
            
        # Draw bounding box
        x, y, w, h = result.bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        
        # Draw center point
        cx, cy = result.iris_center
        cv2.circle(frame, (cx, cy), 2, color, -1)
        
        # Draw iris circle using the calculated radius
        if result.radius is not None:
            cv2.circle(frame, (cx, cy), int(result.radius), color, 1)
        
        # Draw landmarks if requested
        if show_landmarks and result.landmarks is not None:
            for point in result.landmarks:
                px, py = int(point[0]), int(point[1])
                cv2.circle(frame, (px, py), 1, color, -1)
        
        # Draw label and confidence
        label_text = f"{label} Iris" if label else "Iris"
        cv2.putText(frame, f"{label_text}: {result.confidence:.2f}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def reset_history(self):
        """Reset the detection history for both eyes"""
        self.detection_history = {
            "left": [],
            "right": []
        }


def run_iris_detection(camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)) -> bool:
    """
    Run a comprehensive iris detection demo using the webcam.
    
    Args:
        camera_id (int): The camera device ID.
        resolution (Tuple[int, int]): Desired frame resolution.
    
    Returns:
        bool: True if the demo runs successfully, False otherwise.
    """
    logger.info(f"Starting enhanced iris detection demo using camera {camera_id} with resolution {resolution}.")
    
    # Initialize the iris detector with ellipse fitting
    iris_detector = IrisDetector(use_ellipse_fit=True)
    
    # Performance metrics
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    # Detection statistics
    detection_stats = {
        "total_frames": 0,
        "face_detected": 0,
        "left_iris_detected": 0,
        "right_iris_detected": 0
    }
    
    # Create output directory for images
    output_dir = "iris_detection_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create named windows and position them
    main_window = "Iris Detection"
    debug_window = "Debug View"
    cv2.namedWindow(main_window)
    cv2.namedWindow(debug_window)
    cv2.moveWindow(main_window, 50, 50)
    cv2.moveWindow(debug_window, resolution[0] + 80, 50)
    
    try:
        # Initialize camera with retry logic
        retries = 0
        max_retries = 3
        cap = None
        
        while retries < max_retries:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                break
                
            retries += 1
            logger.warning(f"Failed to open camera. Retry {retries}/{max_retries}")
            time.sleep(1)
        
        if not cap.isOpened():
            logger.error(f"Error: Could not open camera (device ID: {camera_id}) after {max_retries} attempts.")
            return False
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera initialized with actual resolution: {actual_width}x{actual_height}")
        
        # Initialize face detector
        from core.face import FaceMeshDetector
        detector = FaceMeshDetector()
        
        # Processing parameters
        show_landmarks = True
        enable_smoothing = True
        show_debug_view = True
        paused = False
        last_frame = None
        
        # Main processing loop
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame from camera. Attempting to reconnect...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        logger.error("Failed to reconnect to camera.")
                        break
                    continue
                
                # Save for pause mode
                last_frame = frame.copy()
            else:
                # Use last captured frame in pause mode
                if last_frame is None:
                    continue
                frame = last_frame.copy()
            
            # Resize frame to target resolution
            frame = cv2.resize(frame, resolution)
            detection_stats["total_frames"] += 1
            
            # Create a debug view showing eye regions
            debug_view = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
            # Process frame to extract facial landmarks
            detection_result = detector.process_frame(frame)
            
            # Update performance metrics
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process detection results
            if detection_result is not None:
                detection_stats["face_detected"] += 1
                
                # Draw face landmarks and bounding box
                detector.draw_landmarks(frame, detection_result.landmarks)
                detector.draw_face_bounding_box(frame, detection_result)
                
                # Detect iris for both eyes
                left_iris_result = iris_detector.detect_iris(detection_result.landmarks, side="left")
                right_iris_result = iris_detector.detect_iris(detection_result.landmarks, side="right")
                
                # Apply smoothing if enabled
                if enable_smoothing:
                    left_smoothed = iris_detector.get_smoothed_result("left")
                    right_smoothed = iris_detector.get_smoothed_result("right")
                    
                    if left_smoothed is not None:
                        left_iris_result = left_smoothed
                    if right_smoothed is not None:
                        right_iris_result = right_smoothed
                
                # Update detection statistics
                if left_iris_result is not None:
                    detection_stats["left_iris_detected"] += 1
                if right_iris_result is not None:
                    detection_stats["right_iris_detected"] += 1
                
                # Draw iris detection results
                if left_iris_result is not None:
                    iris_detector.draw_iris_detection(frame, left_iris_result, 
                                                    color=(255, 0, 0),  # Red for left eye
                                                    show_landmarks=show_landmarks,
                                                    label="Left")
                    
                    # Create zoomed view for debug window
                    if show_debug_view:
                        x, y, w, h = left_iris_result.bounding_box
                        # Expand region for better visibility
                        pad = 20
                        x1, y1 = max(0, x - pad), max(0, y - pad)
                        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                        
                        left_eye_region = frame[y1:y2, x1:x2].copy()
                        if left_eye_region.size > 0:
                            # Scale up for better visibility
                            scale_factor = 3
                            left_eye_region = cv2.resize(left_eye_region, 
                                                       (left_eye_region.shape[1] * scale_factor, 
                                                        left_eye_region.shape[0] * scale_factor))
                            
                            # Calculate position in debug view
                            h_offset = 50
                            if left_eye_region.shape[0] <= debug_view.shape[0] - h_offset and \
                               left_eye_region.shape[1] <= debug_view.shape[1]:
                                debug_view[h_offset:h_offset + left_eye_region.shape[0], 
                                         :left_eye_region.shape[1]] = left_eye_region
                
                if right_iris_result is not None:
                    iris_detector.draw_iris_detection(frame, right_iris_result, 
                                                    color=(0, 255, 0),  # Green for right eye
                                                    show_landmarks=show_landmarks,
                                                    label="Right")
                    
                    # Create zoomed view for debug window
                    if show_debug_view:
                        x, y, w, h = right_iris_result.bounding_box
                        # Expand region for better visibility
                        pad = 20
                        x1, y1 = max(0, x - pad), max(0, y - pad)
                        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                        
                        right_eye_region = frame[y1:y2, x1:x2].copy()
                        if right_eye_region.size > 0:
                            # Scale up for better visibility
                            scale_factor = 3
                            right_eye_region = cv2.resize(right_eye_region, 
                                                        (right_eye_region.shape[1] * scale_factor, 
                                                         right_eye_region.shape[0] * scale_factor))
                            
                            # Calculate position in debug view (below left eye)
                            h_offset = 50 + 250  # Position below left eye view
                            if h_offset + right_eye_region.shape[0] <= debug_view.shape[0] and \
                               right_eye_region.shape[1] <= debug_view.shape[1]:
                                debug_view[h_offset:h_offset + right_eye_region.shape[0], 
                                         :right_eye_region.shape[1]] = right_eye_region
            else:
                cv2.putText(frame, "No face detected", (resolution[0]//3, resolution[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add detection statistics to debug view
            cv2.putText(debug_view, "Detection Statistics:", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            face_rate = detection_stats["face_detected"] / detection_stats["total_frames"] * 100
            left_rate = detection_stats["left_iris_detected"] / detection_stats["face_detected"] * 100 \
                        if detection_stats["face_detected"] > 0 else 0
            right_rate = detection_stats["right_iris_detected"] / detection_stats["face_detected"] * 100 \
                         if detection_stats["face_detected"] > 0 else 0
            
            cv2.putText(debug_view, f"Total frames: {detection_stats['total_frames']}", 
                       (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(debug_view, f"Face detection: {face_rate:.1f}%", 
                       (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(debug_view, f"Left iris: {left_rate:.1f}%", 
                       (10, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
            cv2.putText(debug_view, f"Right iris: {right_rate:.1f}%", 
                       (10, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            
            # Show status of features
            status_text = []
            status_text.append(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
            status_text.append(f"Smoothing: {'ON' if enable_smoothing else 'OFF'}")
            status_text.append(f"Status: {'PAUSED' if paused else 'RUNNING'}")
            
            for i, text in enumerate(status_text):
                cv2.putText(debug_view, text, (10, 670 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
            
            # Display control instructions
            controls = [
                "Controls:",
                "q: Quit",
            ]
            
            for i, text in enumerate(controls):
                cv2.putText(debug_view, text, (resolution[0] - 200, 550 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
            
            # Display the frames
            cv2.imshow(main_window, frame)
            if show_debug_view:
                cv2.imshow(debug_window, debug_view)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Exit requested by user.")
                break
                
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return False
    finally:
        # Clean up resources
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'detector' in locals():
            detector.close()
        cv2.destroyAllWindows()
        
        # Log final statistics
        total_runtime = time.time() - start_time
        face_detection_rate = detection_stats["face_detected"] / detection_stats["total_frames"] * 100 \
                             if detection_stats["total_frames"] > 0 else 0
        left_iris_rate = detection_stats["left_iris_detected"] / detection_stats["face_detected"] * 100 \
                        if detection_stats["face_detected"] > 0 else 0
        right_iris_rate = detection_stats["right_iris_detected"] / detection_stats["face_detected"] * 100 \
                         if detection_stats["face_detected"] > 0 else 0
                         
        logger.info(f"Session summary - Total frames: {detection_stats['total_frames']}, " 
                   f"Face detection rate: {face_detection_rate:.1f}%, "
                   f"Left iris rate: {left_iris_rate:.1f}%, "
                   f"Right iris rate: {right_iris_rate:.1f}%")
        logger.info("Resources released successfully.")
    
    return True

if __name__ == "__main__":
    run_iris_detection()