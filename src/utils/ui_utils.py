import cv2
import numpy as np
import time

def create_calibration_display(width, height, point_x, point_y, step, total_steps):
    """
    Create a calibration display with animated target
    
    Args:
        width: Display width
        height: Display height
        point_x: Target x-coordinate
        point_y: Target y-coordinate
        step: Current calibration step
        total_steps: Total calibration steps
        
    Returns:
        Image with calibration display
    """
    # Create black background
    display = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw outer ring
    color_outer = (0, 0, 255)  # Red
    radius_outer = 30
    cv2.circle(display, (point_x, point_y), radius_outer, color_outer, 2)
    
    # Draw middle ring (pulsating)
    pulse = int(15 + 5 * np.sin(time.time() * 5))
    color_middle = (0, 255, 0)  # Green
    radius_middle = pulse
    cv2.circle(display, (point_x, point_y), radius_middle, color_middle, 2)
    
    # Draw center point
    color_center = (255, 0, 0)  # Blue
    radius_center = 3
    cv2.circle(display, (point_x, point_y), radius_center, color_center, -1)
    
    # Draw progress bar
    progress = step / total_steps
    bar_width = int(width * 0.7)
    bar_height = 10
    bar_x = int((width - bar_width) / 2)
    bar_y = height - 40
    
    # Draw background bar
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (100, 100, 100), -1)
    
    # Draw progress
    progress_width = int(bar_width * progress)
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                 (0, 255, 0), -1)
    
    # Draw text instructions
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Look at the target and press SPACE ({step+1}/{total_steps})"
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = int((width - text_size[0]) / 2)
    cv2.putText(display, text, (text_x, bar_y - 15), font, 0.7, (255, 255, 255), 2)
    
    return display

def draw_gaze_overlay(frame, gaze_x, gaze_y, is_tracking):
    """
    Draw gaze visualization overlay on frame
    
    Args:
        frame: Input frame
        gaze_x, gaze_y: Normalized gaze coordinates (0-1)
        is_tracking: Whether tracking is active
        
    Returns:
        Frame with overlay
    """
    h, w = frame.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    x = int(gaze_x * w)
    y = int(gaze_y * h)
    
    # Draw crosshair
    if is_tracking:
        color = (0, 255, 0)  # Green for active tracking
    else:
        color = (0, 165, 255)  # Orange for inactive
    
    # Draw crosshair
    size = 20
    cv2.line(frame, (x - size, y), (x + size, y), color, 2)
    cv2.line(frame, (x, y - size), (x, y + size), color, 2)
    
    # Draw circle
    cv2.circle(frame, (x, y), 8, color, 2)
    
    return frame

def create_dashboard(frame, metrics):
    """
    Create a dashboard with performance metrics
    
    Args:
        frame: Input frame
        metrics: Dictionary of metrics to display
        
    Returns:
        Frame with dashboard
    """
    # Add a dark semi-transparent overlay at the bottom
    h, w = frame.shape[:2]
    overlay = frame.copy()
    dashboard_height = 100
    cv2.rectangle(overlay, (0, h - dashboard_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add metrics text
    y_pos = h - dashboard_height + 25
    for i, (key, value) in enumerate(metrics.items()):
        text = f"{key}: {value}"
        x_pos = 20 + (i * w // len(metrics))
        cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def create_status_panel(width=400, height=300):
    """
    Create a status panel showing system state
    
    Args:
        width: Panel width
        height: Panel height
        
    Returns:
        Status panel image
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add border
    cv2.rectangle(panel, (0, 0), (width-1, height-1), (100, 100, 100), 1)
    
    # Add title
    cv2.putText(panel, "Eye Gaze Tracker - Status", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return panel
