import numpy as np
import cv2
import math
import time
from typing import List, Tuple, Dict, Optional
import pyautogui

class CalibrationManager:
    def __init__(self, screen_width: int, screen_height: int, calibration_points: int = 9):
        """
        Initialize the calibration manager
        
        Args:
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
            calibration_points: Number of calibration points (9, 5, or 3)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_points = calibration_points
        self.current_point = 0
        self.calibration_complete = False
        
        # Calibration data
        self.target_positions = self._generate_calibration_points(calibration_points)
        self.collected_data = []
        
        # Transformation parameters (will be computed after calibration)
        self.transform_matrix = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.error_threshold = 0.05  # Maximum acceptable error for calibration
        
        # For smoothing
        self.smoothing_window = 5
        self.gaze_history_x = []
        self.gaze_history_y = []

    def _generate_calibration_points(self, num_points: int) -> List[Tuple[float, float]]:
        """Generate calibration points in screen coordinates"""
        points = []
        padding_ratio = 0.1  # 10% padding from screen edges
        
        pad_x = self.screen_width * padding_ratio
        pad_y = self.screen_height * padding_ratio
        
        if num_points == 9:
            # 3x3 grid
            for y_pos in [pad_y, self.screen_height/2, self.screen_height - pad_y]:
                for x_pos in [pad_x, self.screen_width/2, self.screen_width - pad_x]:
                    points.append((x_pos, y_pos))
        elif num_points == 5:
            # 5 points (center and corners)
            points = [
                (self.screen_width/2, self.screen_height/2),  # center
                (pad_x, pad_y),  # top-left
                (self.screen_width - pad_x, pad_y),  # top-right
                (pad_x, self.screen_height - pad_y),  # bottom-left
                (self.screen_width - pad_x, self.screen_height - pad_y)  # bottom-right
            ]
        else:
            # 3 points (center, top-left, bottom-right)
            points = [
                (self.screen_width/2, self.screen_height/2),  # center
                (pad_x, pad_y),  # top-left
                (self.screen_width - pad_x, self.screen_height - pad_y)  # bottom-right
            ]
            
        return points
    
    def get_current_target(self) -> Tuple[float, float]:
        """Get the current calibration target position"""
        if self.current_point < len(self.target_positions):
            return self.target_positions[self.current_point]
        return (self.screen_width/2, self.screen_height/2)  # Default to center if done
    
    def collect_data_point(self, predicted_gaze: Tuple[float, float]) -> bool:
        """
        Collect a calibration data point
        
        Args:
            predicted_gaze: Normalized gaze prediction (x, y) from the model
            
        Returns:
            True if collection was successful, False otherwise
        """
        if self.calibration_complete:
            return False
            
        if self.current_point < len(self.target_positions):
            target = self.target_positions[self.current_point]
            
            # Store the mapping between predicted gaze and target screen position
            self.collected_data.append({
                'predicted': predicted_gaze,
                'target': (target[0] / self.screen_width, target[1] / self.screen_height)  # normalize target
            })
            
            # Move to next point
            self.current_point += 1
            
            # Check if calibration is complete
            if self.current_point >= len(self.target_positions):
                self._compute_calibration()
                self.calibration_complete = True
                
            return True
        
        return False
    
    def _compute_calibration(self) -> None:
        """Compute calibration parameters from collected data"""
        if len(self.collected_data) < 3:
            print("Not enough calibration points collected")
            return
        
        # Extract data points
        predicted_points = np.array([d['predicted'] for d in self.collected_data])
        target_points = np.array([d['target'] for d in self.collected_data])
        
        try:
            # Try to compute a perspective transform (homography) for best mapping
            self.transform_matrix, _ = cv2.findHomography(
                predicted_points.reshape(-1, 1, 2),
                target_points.reshape(-1, 1, 2),
                cv2.RANSAC
            )
            
            # If transform computation fails, fall back to simpler scaling + offset
            if self.transform_matrix is None:
                self._compute_simple_calibration()
            
            # Validate calibration quality
            self._validate_calibration()
            
        except Exception as e:
            print(f"Error computing calibration: {e}")
            self._compute_simple_calibration()
    
    def _compute_simple_calibration(self) -> None:
        """Compute a simpler calibration with scale and offset"""
        if len(self.collected_data) < 2:
            return
            
        # Extract data points
        predicted_x = np.array([d['predicted'][0] for d in self.collected_data])
        predicted_y = np.array([d['predicted'][1] for d in self.collected_data])
        target_x = np.array([d['target'][0] for d in self.collected_data])
        target_y = np.array([d['target'][1] for d in self.collected_data])
        
        # Compute scale and offset using linear regression
        # For X coordinates
        A_x = np.vstack([predicted_x, np.ones(len(predicted_x))]).T
        self.scale_x, self.offset_x = np.linalg.lstsq(A_x, target_x, rcond=None)[0]
        
        # For Y coordinates
        A_y = np.vstack([predicted_y, np.ones(len(predicted_y))]).T
        self.scale_y, self.offset_y = np.linalg.lstsq(A_y, target_y, rcond=None)[0]
        
        # Set transform_matrix to None to indicate we're using the simple calibration
        self.transform_matrix = None
    
    def _validate_calibration(self) -> bool:
        """Validate the quality of the calibration"""
        if len(self.collected_data) < 3:
            return False
            
        total_error = 0
        
        for data_point in self.collected_data:
            predicted = np.array(data_point['predicted'])
            target = np.array(data_point['target'])
            
            # Get the mapped point using current calibration
            mapped = self.map_gaze_to_screen(predicted, normalized=True)
            
            # Calculate error (Euclidean distance)
            error = np.sqrt(np.sum((mapped - target)**2))
            total_error += error
        
        avg_error = total_error / len(self.collected_data)
        
        # Accept calibration if error is below threshold
        self.is_calibrated = avg_error < self.error_threshold
        
        print(f"Calibration {'successful' if self.is_calibrated else 'needs improvement'} "
              f"(avg error: {avg_error:.4f})")
        
        return self.is_calibrated
    
    def map_gaze_to_screen(self, gaze_prediction: Tuple[float, float], normalized: bool = False) -> Tuple[float, float]:
        """
        Map predicted gaze coordinates to screen coordinates
        
        Args:
            gaze_prediction: Gaze prediction from model (normalized 0-1)
            normalized: If True, return normalized screen coordinates (0-1) instead of pixels
            
        Returns:
            Screen coordinates either as (x, y) in pixels or normalized (0-1)
        """
        if not self.calibration_complete:
            # If not calibrated, use a simple center-bias mapping
            if normalized:
                return (0.5, 0.5)
            else:
                return (self.screen_width / 2, self.screen_height / 2)
        
        # Add to history for smoothing
        self.gaze_history_x.append(gaze_prediction[0])
        self.gaze_history_y.append(gaze_prediction[1])
        
        # Keep history to the specified window size
        if len(self.gaze_history_x) > self.smoothing_window:
            self.gaze_history_x.pop(0)
            self.gaze_history_y.pop(0)
        
        # Apply smoothing (simple moving average)
        smoothed_x = sum(self.gaze_history_x) / len(self.gaze_history_x)
        smoothed_y = sum(self.gaze_history_y) / len(self.gaze_history_y)
        
        if self.transform_matrix is not None:
            # Apply homography transform
            gaze_point = np.array([[[smoothed_x, smoothed_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(gaze_point, self.transform_matrix)
            tx, ty = transformed[0][0]
            
            # Blend with head pose (75% gaze, 25% head pose)
            # Note: head pose influence is already factored into the model outputs
        else:
            # Apply simple scale and offset
            tx = smoothed_x * self.scale_x + self.offset_x
            ty = smoothed_y * self.scale_y + self.offset_y
        
        # Clamp to [0, 1] range
        tx = max(0.0, min(1.0, tx))
        ty = max(0.0, min(1.0, ty))
        
        if normalized:
            return (tx, ty)
        else:
            # Convert to pixel coordinates
            return (int(tx * self.screen_width), int(ty * self.screen_height))
    
    def draw_calibration_target(self, frame, window_width, window_height):
        """
        Draw the current calibration target on the frame
        
        Args:
            frame: The frame to draw on
            window_width: Width of the displayed window
            window_height: Height of the displayed window
            
        Returns:
            Frame with calibration target drawn
        """
        if self.calibration_complete:
            return frame
            
        # Get target in screen coordinates
        target_x, target_y = self.get_current_target()
        
        # Convert to window coordinates (assuming window might be different size than screen)
        window_x = int(target_x * window_width / self.screen_width)
        window_y = int(target_y * window_height / self.screen_height)
        
        # Draw target (concentric circles)
        cv2.circle(frame, (window_x, window_y), 20, (0, 0, 255), 2)
        cv2.circle(frame, (window_x, window_y), 10, (0, 255, 0), 2)
        cv2.circle(frame, (window_x, window_y), 2, (255, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, f"Look at the target: {self.current_point+1}/{len(self.target_positions)}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
