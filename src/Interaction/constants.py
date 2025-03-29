# constants.py
import numpy as np
from typing import Final, NamedTuple, Dict
from numpy.typing import NDArray

class LandmarkIndices(NamedTuple):
    """Container for facial landmark index groupings"""
    LEFT_EYE: NDArray[np.int32]
    RIGHT_EYE: NDArray[np.int32]
    LEFT_IRIS: NDArray[np.int32]
    RIGHT_IRIS: NDArray[np.int32]
    NOSE: int
    HEAD_REFERENCE: Dict[str, int]

class FaceLandmarks:
    
    # MediaPipe standard indices (468-point model)
    MEDIAPIPE: Final[LandmarkIndices] = LandmarkIndices(
        LEFT_EYE=np.array([362, 385, 387, 263, 373, 380], dtype=np.int32),
        RIGHT_EYE=np.array([33, 160, 158, 133, 153, 144], dtype=np.int32),
        LEFT_IRIS=np.array([474, 475, 476, 477], dtype=np.int32),
        RIGHT_IRIS=np.array([469, 470, 471, 472], dtype=np.int32),
        NOSE=1,
        HEAD_REFERENCE={
            "left_eye": 33,    # Left eye outer corner
            "right_eye": 263,  # Right eye outer corner
            "nose_tip": 4      # Alternative nose reference
        }
    )

    @classmethod
    def validate_indices(cls, config: LandmarkIndices) -> bool:
        """Validate that all indices are within expected ranges"""
        max_mediapipe = 467
        max_dlib = 67
        
        all_indices = np.concatenate([
            config.LEFT_EYE,
            config.RIGHT_EYE,
            config.LEFT_IRIS,
            config.RIGHT_IRIS,
            [config.NOSE],
            list(config.HEAD_REFERENCE.values())
        ])
        
        if config is cls.MEDIAPIPE:
            return np.all(all_indices <= max_mediapipe)
        elif config is cls.DLIB_COMPAT:
            return np.all(all_indices <= max_dlib)
        return False

# Default configuration using MediaPipe indices
DEFAULT: Final[LandmarkIndices] = FaceLandmarks.MEDIAPIPE

# Define indices for EAR calculation
LEFT_EYE_EAR_IDX = np.array([362, 385, 387, 263, 373, 380], dtype=np.int32)  # Example indices for left eye
RIGHT_EYE_EAR_IDX = np.array([33, 160, 158, 133, 153, 144], dtype=np.int32)  # Example indices for right eye