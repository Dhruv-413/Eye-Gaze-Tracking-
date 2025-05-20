import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional
from collections import deque
import math

# 3D model points for head pose estimation
# Standard facial landmark points in 3D space (based on average face)
MODEL_POINTS_68 = np.array([
    (0.0, 0.0, 0.0),                  # Nose tip
    (0.0, -330.0, -65.0),              # Chin
    (-225.0, 170.0, -135.0),           # Left eye left corner
    (225.0, 170.0, -135.0),            # Right eye right corner
    (-150.0, -150.0, -125.0),          # Left mouth corner
    (150.0, -150.0, -125.0)            # Right mouth corner
])

# MediaPipe face mesh indices that correspond to the 3D model points
FACE_MODEL_INDICES = [
    1,    # Nose tip
    199,  # Chin
    33,   # Left eye left corner
    263,  # Right eye right corner
    61,   # Left mouth corner
    291   # Right mouth corner
]

class GazeProcessor:
    def __init__(self, 
                 smoothing_factor: float = 0.7, 
                 blink_threshold: float = 0.2, 
                 saccade_velocity_threshold: float = 50.0,
                 fixation_duration_threshold: float = 0.1,
                 camera_matrix=None, 
                 dist_coeffs=None):
        """
        Initialize the GazeProcessor for advanced gaze processing
        
        Args:
            smoothing_factor: Factor for exponential smoothing (0-1, higher = more smoothing)
            blink_threshold: EAR threshold for blink detection
            saccade_velocity_threshold: Velocity threshold for saccade detection (pixels/sec)
            fixation_duration_threshold: Minimum duration for fixation classification (seconds)
            camera_matrix: Camera intrinsic matrix, if None estimated from frame
            dist_coeffs: Distortion coefficients, if None assumed zero
        """
        # Parameters
        self.smoothing_factor = smoothing_factor
        self.blink_threshold = blink_threshold
        self.saccade_velocity_threshold = saccade_velocity_threshold
        self.fixation_duration_threshold = fixation_duration_threshold
        
        # Head pose estimation
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))
        
        # State
        self.prev_gaze_point = None
        self.prev_gaze_time = None
        self.smoothed_gaze_point = None
        self.gaze_history = deque(maxlen=30)  # Store last 30 gaze points
        self.velocity_history = deque(maxlen=10)  # Store last 10 velocity measurements
        
        # Gaze event state
        self.current_event = "UNKNOWN"  # FIXATION, SACCADE, BLINK, UNKNOWN
        self.fixation_start_time = None
        self.current_fixation_point = None
        
        # Blink state
        self.is_blinking = False
        self.blink_start_time = None
        self.blink_count = 0
        
        # Calibration data
        self.calibration_points = []  # List of (raw_gaze, target) pairs
        self.calibration_model = None  # Will store regression model after calibration
    
    def estimate_head_pose(self, face_landmarks, frame_size) -> Optional[Dict[str, float]]:
        """
        Estimate head pose using solvePnP
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_size: (width, height) of the frame
            
        Returns:
            Dict with pitch, yaw, and roll in radians, or None if estimation fails
        """
        if face_landmarks is None:
            return None
            
        # Get frame dimensions
        h, w = frame_size[:2]
        
        # If camera matrix is not provided, estimate from frame size
        if self.camera_matrix is None:
            # Estimate camera matrix from frame size
            focal_length = w
            center = (w / 2, h / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )
        
        # Extract specific facial landmarks for pose estimation
        image_points = []
        for idx in FACE_MODEL_INDICES:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                image_points.append([landmark.x * w, landmark.y * h])
        
        if len(image_points) != len(MODEL_POINTS_68):
            return None
            
        image_points = np.array(image_points, dtype=np.float64)
        
        # Solve for pose
        try:
            # Find the rotation and translation vectors
            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_POINTS_68,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None
                
            # Convert rotation vector to rotation matrix 
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Fix the matrix dimension issue in hconcat by ensuring proper shapes
            # Create the 4x4 transformation matrix directly instead of using hconcat
            proj_matrix = np.zeros((4, 4), dtype=np.float64)
            proj_matrix[0:3, 0:3] = rotation_mat
            proj_matrix[0:3, 3] = translation_vec.reshape(3)
            proj_matrix[3, 3] = 1.0
            
            # Extract Euler angles using decomposition
            # Note: We're constructing the complete projection matrix without using hconcat
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix[0:3, :])
            
            # Convert Euler angles from degrees to radians
            pitch = math.radians(euler_angles[0])
            yaw = math.radians(euler_angles[1])
            roll = math.radians(euler_angles[2])
            
            return {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "rotation_vec": rotation_vec,
                "translation_vec": translation_vec
            }
        except Exception as e:
            print(f"Head pose estimation error: {e}")
            return None

    # Add new iris tracking methods
    def get_iris_centers(self, face_landmarks, frame_shape) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get iris centers using MediaPipe's iris landmarks
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of the input frame (h, w, c)
            
        Returns:
            Tuple of left and right iris center as numpy arrays (x, y) or (None, None)
        """
        if face_landmarks is None:
            return None, None
            
        h, w = frame_shape[:2]
        
        # Left iris landmarks (474, 475, 476, 477)
        left_ids = [474, 475, 476, 477]
        try:
            left_xs = [face_landmarks.landmark[i].x * w for i in left_ids]
            left_ys = [face_landmarks.landmark[i].y * h for i in left_ids]
            left_center = np.array([np.mean(left_xs), np.mean(left_ys)])
            
            # Right iris landmarks (469, 470, 471, 472)
            right_ids = [469, 470, 471, 472]
            right_xs = [face_landmarks.landmark[i].x * w for i in right_ids]
            right_ys = [face_landmarks.landmark[i].y * h for i in right_ids]
            right_center = np.array([np.mean(right_xs), np.mean(right_ys)])
            
            return left_center, right_center
        except Exception as e:
            print(f"Error getting iris centers: {e}")
            return None, None
    
    def detect_blink_simple(self, face_landmarks, threshold=0.018):
        """
        Simpler blink detection using vertical distance between eyelids
        
        Args:
            face_landmarks: MediaPipe face landmarks
            threshold: Blink detection threshold (smaller = more sensitive)
            
        Returns:
            True if blink detected, False otherwise
        """
        if face_landmarks is None:
            return False
        
        try:
            # Left eye upper and lower lid landmarks
            top_lid = face_landmarks.landmark[145]  # Upper lid
            bot_lid = face_landmarks.landmark[159]  # Lower lid
            
            # Check vertical distance
            if (top_lid.y - bot_lid.y) < threshold:
                if not self.is_blinking:
                    # Start of a new blink
                    self.is_blinking = True
                    self.blink_start_time = time.time()
                return True
            elif self.is_blinking:
                # End of blink
                self.is_blinking = False
                self.blink_count += 1
                
            return False
        except Exception as e:
            print(f"Error in simple blink detection: {e}")
            return False
    
    def iris_based_tracking(self, face_landmarks, frame_shape, tl_calib=None, br_calib=None):
        """
        Get gaze point using iris position in a simpler way
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of the input frame
            tl_calib: Top-left calibration point
            br_calib: Bottom-right calibration point
            
        Returns:
            Normalized gaze coordinates (x, y) in range [0, 1] or None if not detected
        """
        left_center, right_center = self.get_iris_centers(face_landmarks, frame_shape)
        
        if left_center is None or right_center is None:
            return None
        
        # Use the average of both iris centers
        iris_center = (left_center + right_center) / 2
        
        # If we have calibration points, use them for normalization
        if tl_calib is not None and br_calib is not None:
            # Normalize using calibration extremes
            nx = (iris_center[0] - tl_calib[0]) / max(1.0, (br_calib[0] - tl_calib[0]))
            ny = (iris_center[1] - tl_calib[1]) / max(1.0, (br_calib[1] - tl_calib[1]))
        else:
            # Simple normalization based on frame size
            h, w = frame_shape[:2]
            nx = iris_center[0] / w
            ny = iris_center[1] / h
        
        # Ensure values are within [0, 1]
        nx = np.clip(nx, 0.0, 1.0)
        ny = np.clip(ny, 0.0, 1.0) 
        
        return (nx, ny)
    
    def create_pupil_detection_view(self, left_eye_img, right_eye_img):
        """
        Create a visualization of the pupil detection process
        
        Args:
            left_eye_img: Left eye image
            right_eye_img: Right eye image
            
        Returns:
            Visualization image showing pupil detection steps
        """
        if left_eye_img is None or right_eye_img is None:
            return np.zeros((300, 600, 3), dtype=np.uint8)
            
        # Process left eye
        left_gray = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY) if len(left_eye_img.shape) == 3 else left_eye_img.copy()
        
        # Apply CLAHE for illumination normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        left_clahe = clahe.apply(left_gray)
        
        # Apply Gaussian blur
        left_blur = cv2.GaussianBlur(left_clahe, (7, 7), 0)
        
        # Apply thresholding
        _, left_thresh = cv2.threshold(
            left_blur, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Process right eye similarly
        right_gray = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY) if len(right_eye_img.shape) == 3 else right_eye_img.copy()
        right_clahe = clahe.apply(right_gray)
        right_blur = cv2.GaussianBlur(right_clahe, (7, 7), 0)
        _, right_thresh = cv2.threshold(right_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        left_contours, _ = cv2.findContours(left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        right_contours, _ = cv2.findContours(right_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create color versions for visualization with contours
        left_contour_img = cv2.cvtColor(left_thresh, cv2.COLOR_GRAY2BGR)
        right_contour_img = cv2.cvtColor(right_thresh, cv2.COLOR_GRAY2BGR)
        
        # Draw the largest contour
        if left_contours:
            largest_contour = max(left_contours, key=cv2.contourArea)
            cv2.drawContours(left_contour_img, [largest_contour], -1, (0, 255, 0), 2)
            
            # Fit and draw circle
            try:
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(left_contour_img, center, radius, (0, 0, 255), 2)
            except:
                pass
        
        if right_contours:
            largest_contour = max(right_contours, key=cv2.contourArea)
            cv2.drawContours(right_contour_img, [largest_contour], -1, (0, 255, 0), 2)
            
            # Fit and draw circle
            try:
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(right_contour_img, center, radius, (0, 0, 255), 2)
            except:
                pass
        
        # Resize all images to same height
        target_height = 150
        
        # Left eye row
        left_eye_resized = cv2.resize(left_eye_img, (int(target_height * left_eye_img.shape[1] / left_eye_img.shape[0]), target_height))
        left_clahe_resized = cv2.resize(cv2.cvtColor(left_clahe, cv2.COLOR_GRAY2BGR), (int(target_height * left_clahe.shape[1] / left_clahe.shape[0]), target_height))
        left_thresh_resized = cv2.resize(cv2.cvtColor(left_thresh, cv2.COLOR_GRAY2BGR), (int(target_height * left_thresh.shape[1] / left_thresh.shape[0]), target_height))
        left_contour_resized = cv2.resize(left_contour_img, (int(target_height * left_contour_img.shape[1] / left_contour_img.shape[0]), target_height))
        
        # Right eye row
        right_eye_resized = cv2.resize(right_eye_img, (int(target_height * right_eye_img.shape[1] / right_eye_img.shape[0]), target_height))
        right_clahe_resized = cv2.resize(cv2.cvtColor(right_clahe, cv2.COLOR_GRAY2BGR), (int(target_height * right_clahe.shape[1] / right_clahe.shape[0]), target_height))
        right_thresh_resized = cv2.resize(cv2.cvtColor(right_thresh, cv2.COLOR_GRAY2BGR), (int(target_height * right_thresh.shape[1] / right_thresh.shape[0]), target_height))
        right_contour_resized = cv2.resize(right_contour_img, (int(target_height * right_contour_img.shape[1] / right_contour_img.shape[0]), target_height))
        
        # Get max width for each column
        col1_width = max(left_eye_resized.shape[1], right_eye_resized.shape[1])
        col2_width = max(left_clahe_resized.shape[1], right_clahe_resized.shape[1])
        col3_width = max(left_thresh_resized.shape[1], right_thresh_resized.shape[1])
        col4_width = max(left_contour_resized.shape[1], right_contour_resized.shape[1])
        
        # Padding for uniform columns
        def pad_image(img, target_width):
            h, w = img.shape[:2]
            result = np.zeros((h, target_width, 3), dtype=np.uint8)
            result[:, :w, :] = img
            return result
        
        # Pad all images to their column width
        left_eye_resized = pad_image(left_eye_resized, col1_width)
        right_eye_resized = pad_image(right_eye_resized, col1_width)
        left_clahe_resized = pad_image(left_clahe_resized, col2_width)
        right_clahe_resized = pad_image(right_clahe_resized, col2_width)
        left_thresh_resized = pad_image(left_thresh_resized, col3_width)
        right_thresh_resized = pad_image(right_thresh_resized, col3_width)
        left_contour_resized = pad_image(left_contour_resized, col4_width)
        right_contour_resized = pad_image(right_contour_resized, col4_width)
        
        # Concatenate horizontal rows
        top_row = np.hstack([left_eye_resized, left_clahe_resized, left_thresh_resized, left_contour_resized])
        bottom_row = np.hstack([right_eye_resized, right_clahe_resized, right_thresh_resized, right_contour_resized])
        
        # Add labels
        result = np.vstack([top_row, bottom_row])
        cv2.putText(result, "Left Eye", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, "CLAHE", (col1_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, "Threshold", (col1_width + col2_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, "Pupil Detection", (col1_width + col2_width + col3_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(result, "Right Eye", (10, target_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
            
    def calibrate_iris_tracking(self, face_landmarks, frame_shape, corner="top_left"):
        """
        Get iris position for calibration
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of the input frame
            corner: Which corner to calibrate ("top_left" or "bottom_right")
            
        Returns:
            Iris center position as np.array(x, y) or None if not detected
        """
        left_center, right_center = self.get_iris_centers(face_landmarks, frame_shape)
        
        if left_center is None or right_center is None:
            return None
            
        # Use the average of both iris centers
        iris_center = (left_center + right_center) / 2
        return iris_center

    def detect_pupil(self, eye_img) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """
        Detect pupil center and radius using adaptive thresholding
        
        Args:
            eye_img: Grayscale or color image of the eye region
            
        Returns:
            Tuple of (center_x, center_y), radius or (None, None) if detection fails
        """
        if eye_img is None or eye_img.size == 0:
            return None, None
            
        # Convert to grayscale if needed
        if len(eye_img.shape) == 3:
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_img.copy()
        
        # Apply CLAHE for illumination normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply adaptive thresholding to find dark regions (pupil)
        _, thresholded = cv2.threshold(
            blurred, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresholded, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, None
            
        # Find the largest contour (likely to be the pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Skip if contour is too small
        if cv2.contourArea(largest_contour) < 10:
            return None, None
            
        # Fit circle to the contour
        try:
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            return center, radius
        except:
            return None, None
    
    def calculate_eye_aspect_ratio(self, eye_landmarks) -> float:
        """
        Calculate the eye aspect ratio (EAR) to detect blinks
        
        Args:
            eye_landmarks: List of (x, y) coordinates of eye landmarks
            
        Returns:
            Eye aspect ratio value
        """
        if not eye_landmarks or len(eye_landmarks) < 6:
            return 1.0  # Default to open eye if not enough landmarks
            
        # Compute the euclidean distances
        # Vertical
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Horizontal
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # Return the eye aspect ratio
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 1.0
        return ear
    
    def detect_blink(self, left_eye_ear, right_eye_ear) -> bool:
        """
        Detect blink using eye aspect ratio
        
        Args:
            left_eye_ear: Left eye's aspect ratio
            right_eye_ear: Right eye's aspect ratio
            
        Returns:
            True if blink is detected, False otherwise
        """
        avg_ear = (left_eye_ear + right_eye_ear) / 2.0
        
        # Check if we're below the threshold
        is_current_blink = avg_ear < self.blink_threshold
        
        # State transition
        if is_current_blink and not self.is_blinking:
            # Start of a new blink
            self.is_blinking = True
            self.blink_start_time = time.time()
        elif not is_current_blink and self.is_blinking:
            # End of the blink
            self.is_blinking = False
            self.blink_count += 1
            
        return self.is_blinking
    
    def classify_gaze_event(self, gaze_point, timestamp):
        """
        Classify gaze events (fixation, saccade) based on velocity
        
        Args:
            gaze_point: Current gaze point (x, y)
            timestamp: Current timestamp
            
        Returns:
            Event type string: "FIXATION", "SACCADE", or "UNKNOWN"
        """
        if gaze_point is None:
            return "UNKNOWN"
            
        # If this is the first point, initialize
        if self.prev_gaze_point is None or self.prev_gaze_time is None:
            self.prev_gaze_point = gaze_point
            self.prev_gaze_time = timestamp
            return "UNKNOWN"
            
        # Calculate velocity (pixels per second)
        distance = np.linalg.norm(np.array(gaze_point) - np.array(self.prev_gaze_point))
        time_delta = timestamp - self.prev_gaze_time
        
        if time_delta > 0:
            velocity = distance / time_delta
            self.velocity_history.append(velocity)
        else:
            velocity = 0
            
        # Update previous point
        self.prev_gaze_point = gaze_point
        self.prev_gaze_time = timestamp
        
        # Classify based on velocity
        if velocity < self.saccade_velocity_threshold:
            # This could be a fixation
            if self.current_event != "FIXATION":
                # Start of a new fixation
                self.fixation_start_time = timestamp
                self.current_fixation_point = gaze_point
                
            # Check if fixation has lasted long enough
            if timestamp - self.fixation_start_time >= self.fixation_duration_threshold:
                self.current_event = "FIXATION"
            else:
                self.current_event = "UNKNOWN"
        else:
            # This is a saccade
            self.current_event = "SACCADE"
            self.fixation_start_time = None
            
        return self.current_event
    
    def smooth_gaze(self, gaze_point):
        """
        Apply exponential smoothing to gaze points
        
        Args:
            gaze_point: Current raw gaze point (x, y)
            
        Returns:
            Smoothed gaze point (x, y)
        """
        if gaze_point is None:
            return self.smoothed_gaze_point
            
        # Add to history
        self.gaze_history.append(gaze_point)
        
        # If first point, no smoothing needed
        if self.smoothed_gaze_point is None:
            self.smoothed_gaze_point = gaze_point
            return gaze_point
            
        # Apply exponential smoothing
        alpha = 1.0 - self.smoothing_factor
        smoothed_x = alpha * gaze_point[0] + self.smoothing_factor * self.smoothed_gaze_point[0]
        smoothed_y = alpha * gaze_point[1] + self.smoothing_factor * self.smoothed_gaze_point[1]
        
        self.smoothed_gaze_point = (smoothed_x, smoothed_y)
        return self.smoothed_gaze_point
    
    def add_calibration_point(self, raw_gaze, target_point):
        """
        Add a calibration point for regression-based mapping
        
        Args:
            raw_gaze: Raw gaze point (x, y) as predicted by model
            target_point: Target point (x, y) on screen
        """
        self.calibration_points.append((raw_gaze, target_point))
    
    def train_calibration_model(self):
        """Train a linear regression model to map raw gaze to screen coordinates"""
        if len(self.calibration_points) < 5:
            print("Not enough calibration points")
            return False
            
        try:
            # Extract X (raw gaze) and Y (targets)
            X = np.array([p[0] for p in self.calibration_points])
            Y = np.array([p[1] for p in self.calibration_points])
            
            # Calculate transformation parameters
            # For X mapping
            A_x = np.vstack([X[:, 0], X[:, 1], np.ones(len(X))]).T
            self.calib_params_x = np.linalg.lstsq(A_x, Y[:, 0], rcond=None)[0]
            
            # For Y mapping
            A_y = np.vstack([X[:, 0], X[:, 1], np.ones(len(X))]).T
            self.calib_params_y = np.linalg.lstsq(A_y, Y[:, 1], rcond=None)[0]
            
            # Mark as calibrated
            self.calibration_model = True
            return True
            
        except Exception as e:
            print(f"Error training calibration model: {e}")
            return False
    
    def apply_calibration(self, gaze_point):
        """
        Apply calibration transformation to raw gaze point
        
        Args:
            gaze_point: Raw gaze point (x, y)
            
        Returns:
            Calibrated gaze point (x, y)
        """
        if gaze_point is None or not hasattr(self, 'calibration_model') or not self.calibration_model:
            return gaze_point
            
        try:
            # Apply the learned transformation
            x_mapped = self.calib_params_x[0] * gaze_point[0] + self.calib_params_x[1] * gaze_point[1] + self.calib_params_x[2]
            y_mapped = self.calib_params_y[0] * gaze_point[0] + self.calib_params_y[1] * gaze_point[1] + self.calib_params_y[2]
            
            return (x_mapped, y_mapped)
        except:
            return gaze_point
    
    def visualize_gaze(self, frame, gaze_point, pupil_center=None, pupil_radius=None, head_pose=None):
        """
        Visualize gaze information on frame
        
        Args:
            frame: Input frame to draw on
            gaze_point: Gaze point (x, y) in normalized coordinates
            pupil_center: Optional pupil center coordinates
            pupil_radius: Optional pupil radius
            head_pose: Optional head pose dictionary
            
        Returns:
            Frame with visualizations
        """
        h, w = frame.shape[:2]
        
        # Create a copy of the frame
        viz_frame = frame.copy()
        
        # Draw gaze point
        if gaze_point is not None:
            x, y = int(gaze_point[0] * w), int(gaze_point[1] * h)
            cv2.circle(viz_frame, (x, y), 10, (0, 255, 0), -1)
            cv2.circle(viz_frame, (x, y), 12, (0, 0, 0), 2)
        
        # Draw pupil if detected
        if pupil_center is not None and pupil_radius is not None:
            cv2.circle(viz_frame, pupil_center, pupil_radius, (0, 255, 255), 2)
        
        # Draw head pose vector if available
        if head_pose is not None and "rotation_vec" in head_pose and "translation_vec" in head_pose:
            rot_vec = head_pose["rotation_vec"]
            trans_vec = head_pose["translation_vec"]
            
            # Project a 3D point to the image plane
            nose_end_point3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, 
                                                   rot_vec, 
                                                   trans_vec, 
                                                   self.camera_matrix, 
                                                   self.dist_coeffs)
            
            # Draw the pose arrow
            nose_tip = (int(w/2), int(h/2))  # Approximate nose tip position
            nose_end_point2D = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.arrowedLine(viz_frame, nose_tip, nose_end_point2D, (0, 255, 0), 2)
            
        # Draw gaze event type
        event_colors = {
            "FIXATION": (0, 255, 0),  # Green
            "SACCADE": (0, 165, 255),  # Orange
            "BLINK": (0, 0, 255),     # Red
            "UNKNOWN": (128, 128, 128) # Gray
        }
        
        display_event = "BLINK" if self.is_blinking else self.current_event
        cv2.putText(viz_frame, display_event, (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, event_colors.get(display_event, (255, 255, 255)), 2)
                   
        # Draw blink count
        cv2.putText(viz_frame, f"Blinks: {self.blink_count}", (w - 200, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return viz_frame
    
    def process_gaze_data(self, 
                         left_eye_img, 
                         right_eye_img, 
                         left_eye_landmarks, 
                         right_eye_landmarks,
                         face_landmarks, 
                         frame_shape, 
                         predicted_gaze=None,
                         timestamp=None):
        """
        Process eye images and landmarks to estimate gaze
        
        Args:
            left_eye_img: Left eye image
            right_eye_img: Right eye image
            left_eye_landmarks: Left eye landmarks (list of (x,y) tuples)
            right_eye_landmarks: Right eye landmarks (list of (x,y) tuples)
            face_landmarks: Full face landmarks
            frame_shape: Shape of the original frame
            predicted_gaze: Optional model-predicted gaze point (x,y) in normalized coordinates
            timestamp: Current timestamp, if None will use time.time()
            
        Returns:
            Dictionary with processing results
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Results dictionary
        results = {
            "gaze_point": None,
            "smoothed_gaze_point": None,
            "is_blinking": False,
            "gaze_event": "UNKNOWN",
            "head_pose": None,
            "pupil_left": None,
            "pupil_right": None,
            "ear_left": 1.0,
            "ear_right": 1.0
        }
        
        # Calculate eye aspect ratios for blink detection
        if left_eye_landmarks and len(left_eye_landmarks) >= 6:
            results["ear_left"] = self.calculate_eye_aspect_ratio(left_eye_landmarks)
            
        if right_eye_landmarks and len(right_eye_landmarks) >= 6:
            results["ear_right"] = self.calculate_eye_aspect_ratio(right_eye_landmarks)
        
        # Detect blinks
        results["is_blinking"] = self.detect_blink(results["ear_left"], results["ear_right"])
        
        # If blinking, don't update gaze
        if results["is_blinking"]:
            results["gaze_event"] = "BLINK"
            # Use last known smoothed point
            results["gaze_point"] = self.prev_gaze_point
            results["smoothed_gaze_point"] = self.smoothed_gaze_point
            return results
        
        # Estimate head pose
        if face_landmarks:
            results["head_pose"] = self.estimate_head_pose(face_landmarks, frame_shape)
        
        # Detect pupils in eye images
        if left_eye_img is not None and left_eye_img.size > 0:
            pupil_left_center, pupil_left_radius = self.detect_pupil(left_eye_img)
            if pupil_left_center is not None:
                results["pupil_left"] = (pupil_left_center, pupil_left_radius)
        
        if right_eye_img is not None and right_eye_img.size > 0:
            pupil_right_center, pupil_right_radius = self.detect_pupil(right_eye_img)
            if pupil_right_center is not None:
                results["pupil_right"] = (pupil_right_center, pupil_right_radius)
        
        # Use model-predicted gaze if available, otherwise estimate from pupils
        if predicted_gaze is not None:
            gaze_point = predicted_gaze
        else:
            # Simple fallback if predicted gaze not available
            gaze_point = self.estimate_gaze_from_pupils(
                results.get("pupil_left"), 
                results.get("pupil_right"),
                left_eye_img, 
                right_eye_img
            )
        
        results["gaze_point"] = gaze_point
        
        # Apply calibration if available
        if hasattr(self, 'calibration_model') and self.calibration_model and gaze_point is not None:
            calibrated_gaze = self.apply_calibration(gaze_point)
            results["gaze_point"] = calibrated_gaze
        
        # Classify gaze event
        if gaze_point is not None:
            results["gaze_event"] = self.classify_gaze_event(gaze_point, timestamp)
        
        # Smooth gaze point
        if gaze_point is not None:
            results["smoothed_gaze_point"] = self.smooth_gaze(gaze_point)
        
        # Apply head pose influence (optional enhancement)
        if results["head_pose"] is not None and results["gaze_point"] is not None:
            # Get head pose angles
            pitch = results["head_pose"].get("pitch", 0.0)
            yaw = results["head_pose"].get("yaw", 0.0)
            
            # Adjust gaze point based on head pose (subtle effect)
            # This enhances accuracy when head is moving while eyes are fixated
            x, y = results["smoothed_gaze_point"] or results["gaze_point"]
            
            # Yaw affects horizontal gaze - looking left/right
            yaw_influence = np.clip(yaw / (math.pi/4), -0.2, 0.2) * 0.1  # Scale down the effect
            
            # Pitch affects vertical gaze - looking up/down
            pitch_influence = np.clip(pitch / (math.pi/4), -0.2, 0.2) * 0.1  # Scale down the effect
            
            # Apply the adjusted gaze (75% eye, 25% head pose)
            adjusted_x = max(0.0, min(1.0, x - yaw_influence))
            adjusted_y = max(0.0, min(1.0, y - pitch_influence))
            
            results["smoothed_gaze_point"] = (adjusted_x, adjusted_y)
            
        # Calculate accuracy metrics (if we have ground truth)
        if hasattr(self, 'current_target') and self.current_target is not None and results["gaze_point"] is not None:
            tx, ty = self.current_target
            gx, gy = results["gaze_point"]
            results["accuracy_error"] = math.sqrt((tx-gx)**2 + (ty-gy)**2)
        
        # Store additional diagnostic information
        results["timestamp"] = timestamp
        if self.prev_gaze_time is not None:
            results["frame_delay"] = timestamp - self.prev_gaze_time
            
        # Calculate velocity if we have previous points
        if self.prev_gaze_point is not None and gaze_point is not None:
            if timestamp != self.prev_gaze_time:
                distance = np.linalg.norm(np.array(gaze_point) - np.array(self.prev_gaze_point))
                velocity = distance / (timestamp - self.prev_gaze_time)
                results["gaze_velocity"] = velocity
        
        return results
    
    def estimate_gaze_from_pupils(self, pupil_left, pupil_right, left_eye_img, right_eye_img):
        """
        Estimate gaze direction from pupil positions within eye regions
        
        Args:
            pupil_left: Left pupil (center, radius) or None
            pupil_right: Right pupil (center, radius) or None
            left_eye_img: Left eye image
            right_eye_img: Right eye image
            
        Returns:
            Estimated gaze point (x, y) in normalized coordinates or None
        """
        gaze_x, gaze_y = 0.5, 0.5  # Default to center
        valid_pupils = 0
        
        # Process left eye if available
        if pupil_left is not None and left_eye_img is not None:
            left_center, _ = pupil_left
            left_h, left_w = left_eye_img.shape[:2]
            
            # Normalize pupil position within eye region
            if left_w > 0 and left_h > 0:
                left_norm_x = left_center[0] / left_w
                left_norm_y = left_center[1] / left_h
                
                # Adjust to center (0.5, 0.5 is center)
                left_gaze_x = 1.0 - left_norm_x  # Flip X for natural mapping
                left_gaze_y = left_norm_y
                
                gaze_x += left_gaze_x
                gaze_y += left_gaze_y
                valid_pupils += 1
        
        # Process right eye if available
        if pupil_right is not None and right_eye_img is not None:
            right_center, _ = pupil_right
            right_h, right_w = right_eye_img.shape[:2]
            
            # Normalize pupil position within eye region
            if right_w > 0 and right_h > 0:
                right_norm_x = right_center[0] / right_w
                right_norm_y = right_center[1] / right_h
                
                # Adjust to center (0.5, 0.5 is center)
                right_gaze_x = 1.0 - right_norm_x  # Flip X for natural mapping
                right_gaze_y = right_norm_y
                
                gaze_x += right_gaze_x
                gaze_y += right_gaze_y
                valid_pupils += 1
        
        if valid_pupils > 0:
            # Average the gaze coordinates
            gaze_x /= valid_pupils
            gaze_y /= valid_pupils
            
            # Apply limits
            gaze_x = max(0.0, min(1.0, gaze_x))
            gaze_y = max(0.0, min(1.0, gaze_y))
            
            return (gaze_x, gaze_y)
        else:
            return None
