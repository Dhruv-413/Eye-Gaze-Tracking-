import cv2
<<<<<<< HEAD
import time
import logging
from utils.logging_utils import configure_logging

logger = configure_logging("camera_utils.log")

def open_camera(camera_id: int, resolution: tuple, fps: int = 30, autofocus: int = 1, max_retries: int = 3):
    """Opens and configures the camera; returns the VideoCapture object or None."""
    retries = 0
    cap = None
    while retries < max_retries:
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
            logger.info(f"Camera opened with resolution {resolution}")
            return cap
        retries += 1
        logger.warning(f"Failed to open camera (attempt {retries}/{max_retries}).")
        time.sleep(1)
    logger.error(f"Could not open camera (device ID: {camera_id}) after {max_retries} retries.")
    return None
=======
import numpy as np
import time

class CameraCapture:
    def __init__(self, camera_id=0, width=640, height=480, fps=30, max_retries=3):
        """
        Initialize camera capture with specified parameters
        
        Args:
            camera_id: Camera device ID (default: 0 for primary webcam)
            width: Capture width in pixels
            height: Capture height in pixels
            fps: Target frames per second
            max_retries: Maximum number of retries for camera access
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.max_retries = max_retries
        self.cap = None
        self.frame_count = 0
        self.start_time = 0
        self.actual_fps = 0
        self.last_frame = None  # Store last successful frame
        self.retry_count = 0

    def start(self):
        """Start camera capture with retries"""
        for attempt in range(self.max_retries):
            try:
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                
                if not self.cap.isOpened():
                    print(f"Failed to open camera on attempt {attempt+1}/{self.max_retries}")
                    time.sleep(1)  # Wait before retry
                    continue
                    
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Check if camera is working by reading a test frame
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    print(f"Camera opened but read failed on attempt {attempt+1}/{self.max_retries}")
                    self.cap.release()
                    self.cap = None
                    time.sleep(1)  # Wait before retry
                    continue
                
                self.last_frame = test_frame  # Store first frame
                self.frame_count = 0
                self.start_time = time.time()
                self.retry_count = 0  # Reset retry counter on success
                print(f"Camera started successfully: {self.width}x{self.height} @{self.fps}fps")
                return self
                
            except Exception as e:
                print(f"Error starting camera (attempt {attempt+1}/{self.max_retries}): {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(1)  # Wait before retry
        
        # If all retries fail, raise exception
        raise RuntimeError(f"Failed to start camera after {self.max_retries} attempts")

    def read(self):
        """Read a frame from the camera with error recovery"""
        if self.cap is None:
            # Try to restart if camera is unexpectedly closed
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                print(f"Camera not available, attempting restart ({self.retry_count}/{self.max_retries})")
                try:
                    self.start()
                    return self.read()  # Try reading after restart
                except:
                    pass
            
            # Return the last known good frame if restart fails
            if self.last_frame is not None:
                print("Using last known frame")
                return True, self.last_frame.copy()
            return False, None
            
        # Normal operation
        try:
            ret, frame = self.cap.read()
            
            if ret:
                self.last_frame = frame.copy()  # Save successful frame
                self.frame_count += 1
                self.retry_count = 0  # Reset retry counter on success
                
                # Calculate FPS every second
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1.0:
                    self.actual_fps = self.frame_count / elapsed_time
                    if self.frame_count > 100:  # Reset occasionally to keep measurements current
                        self.frame_count = 0
                        self.start_time = time.time()
            else:
                # Handle failed read
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    print(f"Frame read failed, retrying ({self.retry_count}/{self.max_retries})")
                    time.sleep(0.1)
                    # Return last known good frame
                    if self.last_frame is not None:
                        return True, self.last_frame.copy()
                else:
                    print("Maximum read retries reached, restarting camera")
                    self.release()
                    self.start()
                    self.retry_count = 0
            
            return ret, frame
            
        except Exception as e:
            print(f"Error reading from camera: {e}")
            # Return last known good frame on error
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None
    
    def get_fps(self):
        """Return the actual FPS being achieved"""
        return self.actual_fps
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def display_frame(frame, window_name="Camera Feed", wait_key=1):
    """
    Display a frame in a named window
    
    Args:
        frame: The OpenCV frame to display
        window_name: Name of the display window
        wait_key: Time in ms to wait for key press, controls display refresh rate
        
    Returns:
        Key code of any key pressed, or -1 if no key pressed
    """
    cv2.imshow(window_name, frame)
    return cv2.waitKey(wait_key)


def resize_frame(frame, width=None, height=None, scale=None):
    """
    Resize a frame to specified dimensions or by a scale factor
    
    Args:
        frame: Input frame
        width: Target width (if None, calculated from height and aspect ratio)
        height: Target height (if None, calculated from width and aspect ratio)
        scale: Scale factor, overrides width and height if provided
        
    Returns:
        Resized frame
    """
    if frame is None:
        return None
        
    h, w = frame.shape[:2]
    
    if scale is not None:
        return cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    elif width is None and height is None:
        return frame
    elif width is None:
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    return cv2.resize(frame, (width, height))
>>>>>>> V2
