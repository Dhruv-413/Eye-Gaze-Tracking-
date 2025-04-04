#!/usr/bin/env python3
"""
Optimized real-time gaze tracking using webcam input.
This script provides a lightweight implementation focused on performance.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from typing import Dict, Tuple, List, Optional, Union

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class OptimizedGazeTracker:
    """Optimized gaze tracker class for real-time inference."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "head_pose",
        camera_id: int = 0,
        display_scale: float = 1.0,
        target_size: Tuple[int, int] = (64, 64),
        show_landmarks: bool = False,
        enable_recording: bool = False,
        record_path: str = "gaze_recording.mp4"
    ):
        """
        Initialize the optimized gaze tracker.
        
        Args:
            model_path: Path to the trained model
            model_type: Type of model ('standard', 'multi_input', or 'head_pose')
            camera_id: Camera ID to use
            display_scale: Scale factor for display
            target_size: Target size for model input
            show_landmarks: Whether to show face landmarks
            enable_recording: Whether to record the output
            record_path: Path to save the recording
        """
        self.model_path = model_path
        self.model_type = model_type
        self.camera_id = camera_id
        self.display_scale = display_scale
        self.target_size = target_size
        self.show_landmarks = show_landmarks
        self.enable_recording = enable_recording
        self.record_path = record_path
        
        # Load model
        self.load_model()
        
        # Initialize MediaPipe
        self.setup_mediapipe()
        
        # Initialize camera
        self.setup_camera()
        
        # Initialize visualization parameters
        self.gaze_history = []
        self.max_history = 30
        self.heatmap = np.zeros((300, 400), dtype=np.float32)
        self.heatmap_decay = 0.95
        self.fps_history = []
        self.max_fps_history = 30
    
    def load_model(self):
        """Load the gaze tracking model."""
        print(f"Loading model from {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.model.summary()
            
            # Auto-detect model type if not specified
            if self.model_type == "auto":
                self.model_type = "standard"
                
                if isinstance(self.model.input, dict) or isinstance(self.model.input, list):
                    input_names = [layer.name for layer in self.model.inputs]
                    
                    if any('head_pose' in name for name in input_names):
                        self.model_type = "head_pose"
                    elif any('eye' in name for name in input_names):
                        self.model_type = "multi_input"
            
            print(f"Using model type: {self.model_type}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def setup_mediapipe(self):
        """Set up MediaPipe face mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define landmark indices
        self.landmark_indices = {
            # Left eye landmarks
            'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            # Right eye landmarks
            'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            # Face landmarks for pose estimation
            'pose': [1, 33, 61, 199, 263, 291]
        }
    
    def setup_camera(self):
        """Set up the camera for capturing frames."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {self.camera_id}")
        
        # Get camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera opened with resolution {self.width}x{self.height} at {self.fps} FPS")
        
        # Initialize video writer if recording is enabled
        self.writer = None
        if self.enable_recording:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.record_path, fourcc, self.fps, (self.width, self.height))
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for the model.
        
        Args:
            img: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize to target size
        img_resized = cv2.resize(img, self.target_size)
        
        # Convert to RGB if it's a BGR image
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def extract_eye_regions(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        left_eye_pts = landmarks[self.landmark_indices['left_eye']]
        right_eye_pts = landmarks[self.landmark_indices['right_eye']]
        
        # Calculate eye bounding boxes
        left_eye_min = np.min(left_eye_pts[:, :2], axis=0).astype(int)
        left_eye_max = np.max(left_eye_pts[:, :2], axis=0).astype(int)
        right_eye_min = np.min(right_eye_pts[:, :2], axis=0).astype(int)
        right_eye_max = np.max(right_eye_pts[:, :2], axis=0).astype(int)
        
        # Add margin
        eye_margin = 5
        left_eye_min = np.maximum(0, left_eye_min - eye_margin)
        left_eye_max = np.minimum([w, h], left_eye_max + eye_margin)
        right_eye_min = np.maximum(0, right_eye_min - eye_margin)
        right_eye_max = np.minimum([w, h], right_eye_max + eye_margin)
        
        # Extract eye images
        left_eye_img = frame[left_eye_min[1]:left_eye_max[1], left_eye_min[0]:left_eye_max[0]]
        right_eye_img = frame[right_eye_min[1]:right_eye_max[1], right_eye_min[0]:right_eye_max[0]]
        
        return left_eye_img, right_eye_img
    
    def estimate_head_pose(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate head pose (Euler angles) from face landmarks.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ]) / 4.5
        
        # Camera internals
        focal_length = self.width
        center = (self.width / 2, self.height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # No lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Get specific landmarks for pose estimation
        image_points = np.array([
            landmarks[1],    # Nose tip
            landmarks[152],  # Chin
            landmarks[226],  # Left eye left corner
            landmarks[446],  # Right eye right corner
            landmarks[57],   # Left mouth corner
            landmarks[287]   # Right mouth corner
        ])[:, :2]
        
        # Solve for pose
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if not success:
            return 0.0, 0.0, 0.0
        
        # Convert rotation vector to rotation matrix and then to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = euler_angles.flatten()
        
        # Convert to degrees and adjust signs
        pitch = -pitch
        yaw = -yaw
        
        return pitch, yaw, roll
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a frame for gaze prediction.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Dictionary with processing results
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {"success": False, "error": "No face detected"}
        
        # Get face landmarks
        h, w = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in results.multi_face_landmarks[0].landmark
        ])
        
        # Extract face region
        face_min = np.min(landmarks[:, :2], axis=0).astype(int)
        face_max = np.max(landmarks[:, :2], axis=0).astype(int)
        
        # Add margin
        face_size = face_max - face_min
        margin = (face_size * 0.1).astype(int)
        face_min = np.maximum(0, face_min - margin)
        face_max = np.minimum([w, h], face_max + margin)
        
        face_img = frame[face_min[1]:face_max[1], face_min[0]:face_max[0]]
        
        # Process according to model type
        if self.model_type == "standard":
            # Standard model only needs the face image
            face_input = self.preprocess_image(face_img)
            face_input = np.expand_dims(face_input, axis=0)
            prediction = self.model.predict(face_input, verbose=0)[0]
            
        elif self.model_type == "multi_input":
            # Extract and preprocess eye regions
            left_eye_img, right_eye_img = self.extract_eye_regions(frame, landmarks)
            
            # Preprocess all images
            face_input = self.preprocess_image(face_img)
            left_eye_input = self.preprocess_image(left_eye_img)
            right_eye_input = self.preprocess_image(right_eye_img)
            
            # Add batch dimension
            face_input = np.expand_dims(face_input, axis=0)
            left_eye_input = np.expand_dims(left_eye_input, axis=0)
            right_eye_input = np.expand_dims(right_eye_input, axis=0)
            
            # Prepare model inputs
            model_inputs = {
                'face_input': face_input,
                'left_eye_input': left_eye_input,
                'right_eye_input': right_eye_input
            }
            
            # Predict
            prediction = self.model.predict(model_inputs, verbose=0)[0]
            
        elif self.model_type == "head_pose":
            # Estimate head pose
            pitch, yaw, roll = self.estimate_head_pose(landmarks)
            
            # Preprocess face image
            face_input = self.preprocess_image(face_img)
            face_input = np.expand_dims(face_input, axis=0)
            
            # Prepare head pose input
            head_pose_input = np.array([[pitch, yaw, roll]], dtype=np.float32)
            
            # Prepare model inputs
            model_inputs = {
                'face_input': face_input,
                'head_pose_input': head_pose_input
            }
            
            # Predict
            prediction = self.model.predict(model_inputs, verbose=0)[0]
        
        # Clip prediction to [0, 1] range
        prediction = np.clip(prediction, 0, 1)
        
        # Prepare result
        result = {
            "success": True,
            "prediction": prediction,
            "face_box": (face_min[0], face_min[1], face_max[0], face_max[1]),
            "landmarks": landmarks
        }
        
        # Add head pose if needed
        if self.model_type == "head_pose":
            result["head_pose"] = (pitch, yaw, roll)
        
        return result
    
    def visualize_result(self, frame: np.ndarray, result: Dict, fps: float) -> np.ndarray:
        """
        Visualize the result on the frame.
        
        Args:
            frame: Input frame
            result: Processing result
            fps: Current FPS
            
        Returns:
            Annotated frame
        """
        # Create a copy of the frame
        display = frame.copy()
        
        if not result["success"]:
            # Display error
            cv2.putText(
                display, 
                f"Error: {result.get('error', 'Unknown error')}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            return display
        
        # Draw face box
        x1, y1, x2, y2 = result["face_box"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks if requested
        if self.show_landmarks and "landmarks" in result:
            for i, (x, y, _) in enumerate(result["landmarks"]):
                # Draw only a subset of landmarks
                if i % 8 == 0:
                    cv2.circle(display, (int(x), int(y)), 1, (0, 255, 255), -1)
        
        # Convert normalized gaze coordinates to pixel coordinates
        h, w = display.shape[:2]
        pred_x, pred_y = result["prediction"]
        x_pixel = int(pred_x * w)
        y_pixel = int(pred_y * h)
        
        # Draw current gaze point
        cv2.circle(display, (x_pixel, y_pixel), 10, (0, 0, 255), -1)
        
        # Draw gaze history
        for i, point in enumerate(self.gaze_history):
            alpha = (i + 1) / len(self.gaze_history)
            point_x = int(point[0] * w)
            point_y = int(point[1] * h)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            size = max(1, int(5 * alpha))
            cv2.circle(display, (point_x, point_y), size, color, -1)
        
        # Draw FPS
        cv2.putText(
            display, 
            f"FPS: {fps:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Draw gaze coordinates
        cv2.putText(
            display, 
            f"Gaze: ({pred_x:.3f}, {pred_y:.3f})", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Add head pose info if available
        if "head_pose" in result:
            pitch, yaw, roll = result["head_pose"]
            cv2.putText(
                display, 
                f"Pose: P={pitch:.1f}, Y={yaw:.1f}, R={roll:.1f}", 
                (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        return display
    
    def run(self):
        """Run the gaze tracking loop."""
        # Create windows
        cv2.namedWindow("Gaze Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Tracking", int(self.width * self.display_scale), int(self.height * self.display_scale))
        
        # Create heatmap window
        heatmap_size = (400, 300)
        cv2.namedWindow("Gaze Heatmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Heatmap", heatmap_size[0], heatmap_size[1])
        
        print("Running gaze tracker. Press 'q' to quit, 's' to save a screenshot.")
        
        # For FPS calculation
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        current_fps = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame. Exiting.")
                    break
                
                # Process frame
                start_process = time.time()
                result = self.process_frame(frame)
                process_time = time.time() - start_process
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                
                # Update FPS every second
                if elapsed - last_fps_update >= 1.0:
                    current_fps = frame_count / (elapsed - last_fps_update)
                    self.fps_history.append(current_fps)
                    if len(self.fps_history) > self.max_fps_history:
                        self.fps_history.pop(0)
                    frame_count = 0
                    last_fps_update = elapsed
                
                # Show processing time in console occasionally
                if frame_count % 30 == 0:
                    print(f"Processing time: {process_time*1000:.1f}ms, FPS: {current_fps:.1f}")
                
                # Update gaze history and heatmap if detection was successful
                if result["success"]:
                    gaze_point = result["prediction"]
                    
                    # Update gaze history
                    self.gaze_history.append(gaze_point)
                    if len(self.gaze_history) > self.max_history:
                        self.gaze_history.pop(0)
                    
                    # Update heatmap
                    x_heatmap = int(gaze_point[0] * 400)
                    y_heatmap = int(gaze_point[1] * 300)
                    cv2.circle(self.heatmap, (x_heatmap, y_heatmap), 15, (5,), -1)
                
                # Apply decay to heatmap
                self.heatmap *= self.heatmap_decay
                
                # Visualize results
                avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                display = self.visualize_result(frame, result, avg_fps)
                
                # Show frames
                cv2.imshow("Gaze Tracking", display)
                
                # Show heatmap
                heatmap_vis = cv2.applyColorMap(
                    np.uint8(np.clip(self.heatmap * 10, 0, 255)), 
                    cv2.COLORMAP_JET
                )
                cv2.rectangle(heatmap_vis, (0, 0), (399, 299), (0, 255, 0), 2)
                cv2.imshow("Gaze Heatmap", heatmap_vis)
                
                # Record if enabled
                if self.writer is not None:
                    self.writer.write(display)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"gaze_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display)
                    print(f"Screenshot saved to {filename}")
        
        finally:
            # Clean up
            self.cap.release()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            print("Gaze tracker stopped.")

def main():
    parser = argparse.ArgumentParser(description="Optimized real-time gaze tracking")
    
    # Model options
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", type=str, default="auto",
                      choices=["auto", "standard", "multi_input", "head_pose"],
                      help="Type of model (auto=detect automatically)")
    
    # Camera options
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--scale", type=float, default=1.0, help="Display scale factor")
    parser.add_argument("--show_landmarks", action="store_true", help="Show face landmarks")
    
    # Recording options
    parser.add_argument("--record", action="store_true", help="Record video output")
    parser.add_argument("--record_path", type=str, default="gaze_recording.mp4", 
                      help="Path for recorded video")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    try:
        # Create and run tracker
        tracker = OptimizedGazeTracker(
            model_path=args.model,
            model_type=args.model_type,
            camera_id=args.camera,
            display_scale=args.scale,
            show_landmarks=args.show_landmarks,
            enable_recording=args.record,
            record_path=args.record_path
        )
        
        tracker.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
