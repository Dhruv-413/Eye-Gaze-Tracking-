#!/usr/bin/env python3
"""
Real-time gaze tracking using a webcam.
This script loads a trained gaze tracking model and performs inference using the webcam.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from typing import Dict, Tuple, List, Optional, Union

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.gaze_tracking.visualization import visualize_head_pose

class RealtimeGazeTracker:
    def __init__(
        self, 
        model_path: str,
        model_type: str = 'standard',
        use_mediapipe: bool = True,
        camera_id: int = 0,
        display_scale: float = 1.0,
        show_head_pose: bool = False
    ):
        """
        Initialize the gaze tracker with a trained model.
        
        Args:
            model_path: Path to the trained model (.h5)
            model_type: Type of model ('standard', 'multi_input', or 'head_pose')
            use_mediapipe: Whether to use MediaPipe for face/eye detection
            camera_id: Camera ID to use (default: 0)
            display_scale: Scale factor for display (default: 1.0)
            show_head_pose: Whether to show head pose visualization
        """
        self.model_path = model_path
        self.model_type = model_type
        self.camera_id = camera_id
        self.display_scale = display_scale
        self.show_head_pose = show_head_pose
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera with ID {camera_id}")
        
        # Get camera resolution
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
        
        # Initialize face detection
        if use_mediapipe:
            self.init_mediapipe()
        else:
            self.init_opencv_face_detector()
            
        # Load model
        print(f"Loading model from {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model.summary()
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
            
        # Get model input shape
        if model_type == 'standard':
            self.input_shape = self.model.input_shape[1:3]
        else:
            # For multi-input models, get the shape from the face input
            self.input_shape = self.model.get_layer('face_input').input_shape[1:3]
            
        print(f"Model input shape: {self.input_shape}")
        
        # Frame processing stats
        self.fps_history = []
        self.max_fps_history = 30
        
        # Initialize visualization
        self.init_visualization()
    
    def init_mediapipe(self):
        """Initialize MediaPipe face mesh for face and eye detection."""
        self.use_mediapipe = True
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define eye landmarks (taken from MediaPipe documentation)
        # Left eye landmarks
        self.left_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right eye landmarks
        self.right_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Face landmark indices for head pose estimation
        self.face_landmarks_for_pose = [33, 263, 1, 61, 291, 199]
        
    def init_opencv_face_detector(self):
        """Initialize OpenCV's face and eye detectors as a fallback."""
        self.use_mediapipe = False
        
        # Load face cascade
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(face_cascade_path):
            raise FileNotFoundError(f"Face cascade not found at {face_cascade_path}")
        
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Load eye cascade
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        if not os.path.exists(eye_cascade_path):
            raise FileNotFoundError(f"Eye cascade not found at {eye_cascade_path}")
            
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    def init_visualization(self):
        """Initialize visualization windows and parameters."""
        # Heatmap accumulator and parameters
        self.heatmap_size = (400, 300)
        self.heatmap = np.zeros((self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        self.heatmap_decay = 0.95  # Decay factor for heatmap
        self.gaze_history = []
        self.max_gaze_history = 30  # Max points to show in trail
        
        # Create named window for gaze heatmap
        cv2.namedWindow("Gaze Heatmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Heatmap", self.heatmap_size[0], self.heatmap_size[1])
        
        # Create main window for video feed
        cv2.namedWindow("Gaze Tracking", cv2.WINDOW_NORMAL)
        
        # Calculate display size
        display_width = int(self.frame_width * self.display_scale)
        display_height = int(self.frame_height * self.display_scale)
        cv2.resizeWindow("Gaze Tracking", display_width, display_height)
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for the model.
        
        Args:
            face_img: Face image
            
        Returns:
            Preprocessed face image
        """
        # Resize to model input shape
        face_img = cv2.resize(face_img, self.input_shape)
        
        # Convert to RGB if it's a BGR image
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_img = face_img.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    
    def preprocess_eye(self, eye_img: np.ndarray) -> np.ndarray:
        """
        Preprocess eye image for the model.
        
        Args:
            eye_img: Eye image
            
        Returns:
            Preprocessed eye image
        """
        # Resize to model input shape
        eye_img = cv2.resize(eye_img, self.input_shape)
        
        # Convert to RGB if it's a BGR image
        if len(eye_img.shape) == 3 and eye_img.shape[2] == 3:
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        eye_img = eye_img.astype(np.float32) / 255.0
        
        # Add batch dimension
        eye_img = np.expand_dims(eye_img, axis=0)
        
        return eye_img
    
    def get_landmarks(self, results, img_shape) -> Optional[np.ndarray]:
        """
        Extract face landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe face mesh results
            img_shape: Input image shape (height, width)
            
        Returns:
            Normalized landmarks as numpy array or None if no face detected
        """
        if not results.multi_face_landmarks:
            return None
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        landmarks = np.array([
            [lm.x * img_shape[1], lm.y * img_shape[0], lm.z * img_shape[1]]
            for lm in face_landmarks.landmark
        ])
        
        return landmarks
    
    def extract_eye_regions(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract left and right eye regions based on landmarks.
        
        Args:
            frame: Input frame
            landmarks: Face landmarks
            
        Returns:
            Tuple of (left_eye_img, right_eye_img)
        """
        h, w = frame.shape[:2]
        
        # Get eye landmarks
        left_eye_pts = landmarks[self.left_eye_landmarks]
        right_eye_pts = landmarks[self.right_eye_landmarks]
        
        # Calculate bounding box for each eye
        left_eye_min = np.min(left_eye_pts, axis=0).astype(int)
        left_eye_max = np.max(left_eye_pts, axis=0).astype(int)
        right_eye_min = np.min(right_eye_pts, axis=0).astype(int)
        right_eye_max = np.max(right_eye_pts, axis=0).astype(int)
        
        # Add margin (20% of eye size)
        left_eye_size = left_eye_max - left_eye_min
        right_eye_size = right_eye_max - right_eye_min
        left_margin = (left_eye_size * 0.2).astype(int)
        right_margin = (right_eye_size * 0.2).astype(int)
        
        left_eye_min = np.maximum(0, left_eye_min - left_margin)
        left_eye_max = np.minimum([w, h, 0], left_eye_max + left_margin)
        right_eye_min = np.maximum(0, right_eye_min - right_margin)
        right_eye_max = np.minimum([w, h, 0], right_eye_max + right_margin)
        
        # Extract eye regions
        left_eye_img = frame[left_eye_min[1]:left_eye_max[1], left_eye_min[0]:left_eye_max[0]]
        right_eye_img = frame[right_eye_min[1]:right_eye_max[1], right_eye_min[0]:right_eye_max[0]]
        
        # Check if valid regions were extracted
        if left_eye_img.size == 0 or right_eye_img.size == 0:
            return None, None
        
        return left_eye_img, right_eye_img
    
    def estimate_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Estimate head pose (Euler angles) from face landmarks.
        
        Args:
            landmarks: Face landmarks
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # 3D model points (standard face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ]) / 4.5  # Scale down for better visualization
        
        # Camera internals
        h, w = frame_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Get specific landmarks for pose estimation
        image_points = np.array([
            landmarks[1],    # Nose tip
            landmarks[152],  # Chin
            landmarks[226],  # Left eye left corner
            landmarks[446],  # Right eye right corner
            landmarks[57],   # Left mouth corner
            landmarks[287]   # Right mouth corner
        ])[:, :2]  # Keep only x,y
        
        # Solve for pose
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if not success:
            return 0.0, 0.0, 0.0
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Convert to Euler angles
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = euler_angles.flatten()
        
        # Convert to degrees and adjust signs
        pitch = -pitch
        yaw = -yaw
        
        return pitch, yaw, roll
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> Dict:
        """
        Detect faces using MediaPipe face mesh.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary containing detection results
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {"success": False}
        
        # Get landmarks as numpy array
        h, w = frame.shape[:2]
        landmarks = self.get_landmarks(results, (h, w))
        
        if landmarks is None:
            return {"success": False}
        
        # Extract face region
        face_min = np.min(landmarks[:, :2], axis=0).astype(int)
        face_max = np.max(landmarks[:, :2], axis=0).astype(int)
        
        # Add margin (20% of face size)
        face_size = face_max - face_min
        margin = (face_size * 0.1).astype(int)
        face_min = np.maximum(0, face_min - margin)
        face_max = np.minimum([w, h], face_max + margin)
        
        face_img = frame[face_min[1]:face_max[1], face_min[0]:face_max[0]]
        
        # Extract eye regions
        left_eye_img, right_eye_img = self.extract_eye_regions(frame, landmarks)
        
        # Estimate head pose
        pitch, yaw, roll = self.estimate_head_pose(landmarks, (h, w))
        
        return {
            "success": True,
            "landmarks": landmarks,
            "face": face_img,
            "left_eye": left_eye_img,
            "right_eye": right_eye_img,
            "face_box": (face_min[0], face_min[1], face_max[0], face_max[1]),
            "head_pose": (pitch, yaw, roll)
        }
    
    def detect_faces_opencv(self, frame: np.ndarray) -> Dict:
        """
        Detect faces using OpenCV's Haar cascades.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary containing detection results
        """
        # Convert to grayscale for cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return {"success": False}
        
        # Get the largest face
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]
        
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        face_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_gray)
        
        left_eye_img = None
        right_eye_img = None
        
        if len(eyes) >= 2:
            # Sort by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Left eye (in the image, not anatomically)
            ex, ey, ew, eh = eyes[0]
            left_eye_img = face_img[ey:ey+eh, ex:ex+ew]
            
            # Right eye (in the image, not anatomically)
            ex, ey, ew, eh = eyes[1]
            right_eye_img = face_img[ey:ey+eh, ex:ex+ew]
        
        # No head pose estimation with OpenCV
        return {
            "success": True,
            "face": face_img,
            "left_eye": left_eye_img,
            "right_eye": right_eye_img,
            "face_box": (x, y, x+w, y+h),
            "head_pose": (0.0, 0.0, 0.0)  # Default neutral pose
        }
    
    def update_display(self, frame: np.ndarray, gaze_point: Tuple[float, float], 
                      detection_results: Dict, processing_time: float) -> np.ndarray:
        """
        Update the display with gaze point and detection info.
        
        Args:
            frame: Input frame
            gaze_point: Predicted gaze point (x, y) normalized to [0, 1]
            detection_results: Face detection results
            processing_time: Processing time for the frame in seconds
            
        Returns:
            Annotated frame
        """
        # Create a copy of the frame
        display = frame.copy()
        
        # Calculate FPS
        fps = 1.0 / processing_time if processing_time > 0 else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_fps_history:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Add gaze point to history
        self.gaze_history.append(gaze_point)
        if len(self.gaze_history) > self.max_gaze_history:
            self.gaze_history.pop(0)
        
        # Draw face box if detection was successful
        if detection_results["success"]:
            x1, y1, x2, y2 = detection_results["face_box"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert normalized gaze coordinates to pixel coordinates
        h, w = display.shape[:2]
        x_pixel = int(gaze_point[0] * w)
        y_pixel = int(gaze_point[1] * h)
        
        # Draw gaze point
        cv2.circle(display, (x_pixel, y_pixel), 10, (0, 0, 255), -1)
        
        # Draw gaze history (trail)
        for i, point in enumerate(self.gaze_history[:-1]):
            alpha = (i + 1) / len(self.gaze_history)  # Opacity increases with newer points
            point_x = int(point[0] * w)
            point_y = int(point[1] * h)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            size = max(1, int(5 * alpha))
            cv2.circle(display, (point_x, point_y), size, color, -1)
        
        # Add FPS text
        cv2.putText(
            display, f"FPS: {avg_fps:.1f}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Add head pose info if available
        if self.model_type in ['multi_input', 'head_pose'] and detection_results["success"]:
            pitch, yaw, roll = detection_results["head_pose"]
            cv2.putText(
                display, f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Update heatmap
        self.heatmap *= self.heatmap_decay  # Apply decay
        x_heatmap = int(gaze_point[0] * self.heatmap_size[0])
        y_heatmap = int(gaze_point[1] * self.heatmap_size[1])
        cv2.circle(
            self.heatmap, 
            (x_heatmap, y_heatmap), 
            15, 
            (5,), 
            -1
        )
        
        # Normalize heatmap for visualization
        heatmap_vis = cv2.applyColorMap(
            np.uint8(np.clip(self.heatmap * 10, 0, 255)), 
            cv2.COLORMAP_JET
        )
        
        # Draw screen boundaries on heatmap
        cv2.rectangle(
            heatmap_vis, 
            (0, 0), 
            (self.heatmap_size[0] - 1, self.heatmap_size[1] - 1), 
            (0, 255, 0), 
            2
        )
        
        # Show heatmap in separate window
        cv2.imshow("Gaze Heatmap", heatmap_vis)
        
        return display
    
    def predict_gaze(self, detection_results: Dict) -> Tuple[float, float]:
        """
        Predict gaze point based on the model type.
        
        Args:
            detection_results: Face detection results
            
        Returns:
            Tuple of (x, y) coordinates normalized to [0, 1]
        """
        if not detection_results["success"]:
            return (0.5, 0.5)  # Default to center if no face detected
        
        # Standard model (face only)
        if self.model_type == 'standard':
            face_input = self.preprocess_face(detection_results["face"])
            prediction = self.model.predict(face_input, verbose=0)[0]
            
        # Multi-input model (face and eyes)
        elif self.model_type == 'multi_input':
            if detection_results["left_eye"] is None or detection_results["right_eye"] is None:
                return (0.5, 0.5)  # Default to center if eyes not detected
                
            face_input = self.preprocess_face(detection_results["face"])
            left_eye_input = self.preprocess_eye(detection_results["left_eye"])
            right_eye_input = self.preprocess_eye(detection_results["right_eye"])
            
            model_inputs = {
                'face_input': face_input,
                'left_eye_input': left_eye_input,
                'right_eye_input': right_eye_input
            }
            
            prediction = self.model.predict(model_inputs, verbose=0)[0]
            
        # Head pose model (face and head pose)
        elif self.model_type == 'head_pose':
            face_input = self.preprocess_face(detection_results["face"])
            pitch, yaw, roll = detection_results["head_pose"]
            head_pose_input = np.array([[pitch, yaw, roll]], dtype=np.float32)
            
            model_inputs = {
                'face_input': face_input,
                'head_pose_input': head_pose_input
            }
            
            prediction = self.model.predict(model_inputs, verbose=0)[0]
        
        # Clip to [0, 1] range
        prediction = np.clip(prediction, 0, 1)
        
        return tuple(prediction)
    
    def run(self):
        """Run the real-time gaze tracking loop."""
        print("Starting real-time gaze tracking. Press 'q' to quit.")
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame. Exiting.")
                    break
                
                # Detect face(s)
                if self.use_mediapipe:
                    detection_results = self.detect_faces_mediapipe(frame)
                else:
                    detection_results = self.detect_faces_opencv(frame)
                
                # If detection succeeded, predict gaze
                if detection_results["success"]:
                    gaze_point = self.predict_gaze(detection_results)
                else:
                    gaze_point = (0.5, 0.5)  # Default to center
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Update display
                display_frame = self.update_display(frame, gaze_point, detection_results, processing_time)
                
                # Show the frame
                cv2.imshow("Gaze Tracking", display_frame)
                
                # Show head pose visualization if requested
                if self.show_head_pose and detection_results["success"]:
                    visualize_head_pose(
                        detection_results["face"], 
                        detection_results["head_pose"],
                        save_path=None
                    )
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save the current frame
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f"gaze_tracking_{timestamp}.jpg", display_frame)
                    print(f"Saved frame to gaze_tracking_{timestamp}.jpg")
                
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            if self.use_mediapipe:
                self.face_mesh.close()

def main():
    parser = argparse.ArgumentParser(description="Real-time gaze tracking")
    
    # Model parameters
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", type=str, default="standard", 
                      choices=["standard", "multi_input", "head_pose"],
                      help="Type of model to use")
    
    # Camera parameters
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--scale", type=float, default=1.0, help="Display scale factor")
    
    # Detection parameters
    parser.add_argument("--use_opencv", action="store_true", 
                      help="Use OpenCV instead of MediaPipe for detection")
    
    # Visualization parameters
    parser.add_argument("--show_head_pose", action="store_true", 
                      help="Show 3D head pose visualization")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    try:
        tracker = RealtimeGazeTracker(
            model_path=args.model,
            model_type=args.model_type,
            use_mediapipe=not args.use_opencv,
            camera_id=args.camera,
            display_scale=args.scale,
            show_head_pose=args.show_head_pose
        )
        
        tracker.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
