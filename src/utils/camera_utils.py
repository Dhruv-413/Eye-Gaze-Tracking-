import cv2
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
