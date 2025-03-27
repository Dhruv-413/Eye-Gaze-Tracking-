# core/utils.py
import cv2
import numpy as np
from typing import Tuple, Sequence

def calculate_ear(eye: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for blink detection.

    Args:
        eye (np.ndarray): Array of 6 (x, y) points representing an eye.

    Returns:
        float: Computed EAR value.
    """
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, color: Tuple[int, int, int]=(0, 255, 0)) -> None:
    """
    Draw landmarks on the frame.

    Args:
        frame (np.ndarray): The image to draw on.
        landmarks (np.ndarray): A sequence of (x, y) landmark points.
        color (tuple): BGR color tuple.
    """
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 1, color, -1)
