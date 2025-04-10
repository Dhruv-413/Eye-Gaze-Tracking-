import cv2
import numpy as np
import time

class CameraCapture:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Initialize camera capture with specified parameters
        
        Args:
            camera_id: Camera device ID (default: 0 for primary webcam)
            width: Capture width in pixels
            height: Capture height in pixels
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame_count = 0
        self.start_time = 0
        self.actual_fps = 0

    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID {self.camera_id}")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Reset metrics
        self.frame_count = 0
        self.start_time = time.time()
        return self

    def read(self):
        """Read a frame from the camera"""
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() first.")
            
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1.0:  # Update FPS calculation every second
                self.actual_fps = self.frame_count / elapsed_time
        
        return ret, frame

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
