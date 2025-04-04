import numpy as np

def calculate_ear(eye_landmarks):
    """Calculate the Eye Aspect Ratio (EAR)."""
    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
    height1 = np.linalg.norm(p2 - p6)
    height2 = np.linalg.norm(p3 - p5)
    width = np.linalg.norm(p1 - p4)
    return (height1 + height2) / (2.0 * width) if width > 0 else 0.0

def normalize_coordinates(landmarks, width, height):
    """Normalize landmarks to the given width and height."""
    return landmarks * np.array([width, height])
