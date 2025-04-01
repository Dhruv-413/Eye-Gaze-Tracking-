# utils.py
import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 0)
DEFAULT_RADIUS: int = 1
DEFAULT_LINE_TYPE: int = cv2.LINE_AA
DEFAULT_THICKNESS: int = 1
DEFAULT_EPSILON: float = 1e-6

class InvalidInputError(Exception):
    """Exception raised for invalid inputs."""
    pass

def validate_array(arr: NDArray, expected_dim: int, expected_shape: Optional[Tuple[int, ...]] = None, arr_name: str = "Array") -> None:
    if not isinstance(arr, np.ndarray):
        error_msg = f"{arr_name} must be a numpy array, got {type(arr)}."
        logger.error(error_msg)
        raise InvalidInputError(error_msg)
    if arr.ndim != expected_dim:
        error_msg = f"{arr_name} must be {expected_dim}D, got {arr.ndim}D."
        logger.error(error_msg)
        raise InvalidInputError(error_msg)
    if expected_shape is not None and arr.shape != expected_shape:
        error_msg = f"{arr_name} must have shape {expected_shape}, got {arr.shape}."
        logger.error(error_msg)
        raise InvalidInputError(error_msg)
    if arr.size == 0:
        error_msg = f"{arr_name} is empty."
        logger.error(error_msg)
        raise InvalidInputError(error_msg)
    if np.isnan(arr).any():
        error_msg = f"{arr_name} contains NaN values."
        logger.error(error_msg)
        raise InvalidInputError(error_msg)

def calculate_ear(eye: NDArray[np.float32], epsilon: float = DEFAULT_EPSILON) -> float:
    logger.debug("Calculating EAR.")
    validate_array(eye, expected_dim=2, expected_shape=(6, 2), arr_name="eye")
    try:
        vertical1 = np.linalg.norm(eye[1] - eye[5])
        vertical2 = np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        ear_value = (vertical1 + vertical2) / (2.0 * horizontal + epsilon)
        logger.debug("EAR calculated: %.4f", ear_value)
        return ear_value
    except Exception as error:
        logger.error("Error in calculate_ear: %s", error, exc_info=True)
        raise

def draw_landmarks(
    frame: NDArray[np.uint8],
    landmarks: NDArray[np.float32],
    color: Tuple[int, int, int] = DEFAULT_COLOR,
    radius: int = DEFAULT_RADIUS,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE
) -> None:
    logger.debug("Drawing landmarks.")
    validate_array(frame, expected_dim=3, arr_name="frame")
    validate_array(landmarks, expected_dim=2, arr_name="landmarks")
    if landmarks.shape[1] != 2:
        error_msg = f"landmarks must have shape (N, 2), got {landmarks.shape}."
        logger.error(error_msg)
        raise InvalidInputError(error_msg)
    try:
        pts = landmarks.astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness)
        for (x, y) in pts:
            cv2.circle(frame, (x, y), radius, color, -1, lineType=line_type)
        logger.debug("Landmarks drawn successfully.")
    except Exception as error:
        logger.error("Error in draw_landmarks: %s", error, exc_info=True)
        raise

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO, console: bool = True) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Logging initialized.")
