# constants.py
import numpy as np
from typing import Final, NamedTuple, Dict
from numpy.typing import NDArray

MAX_MEDIAPIPE_INDEX: Final[int] = 467
MAX_DLIB_INDEX: Final[int] = 67

class LandmarkIndices(NamedTuple):
    """Container for facial landmark index groupings."""
    LEFT_EYE: NDArray[np.int32]
    RIGHT_EYE: NDArray[np.int32]
    LEFT_IRIS: NDArray[np.int32]
    RIGHT_IRIS: NDArray[np.int32]
    NOSE: int
    HEAD_REFERENCE: Dict[str, int]

class FaceLandmarks:
    """
    Provides standard facial landmark index configurations.
    Supports MediaPipe and an example DLIB configuration.
    """
    MEDIAPIPE: Final[LandmarkIndices] = LandmarkIndices(
        LEFT_EYE=np.array([362, 385, 387, 263, 373, 380], dtype=np.int32),
        RIGHT_EYE=np.array([33, 160, 158, 133, 153, 144], dtype=np.int32),
        LEFT_IRIS=np.array([474, 475, 476, 477], dtype=np.int32),
        RIGHT_IRIS=np.array([469, 470, 471, 472], dtype=np.int32),
        NOSE=1,
        HEAD_REFERENCE={"left_eye": 33, "right_eye": 263, "nose_tip": 4}
    )

    DLIB_COMPAT: Final[LandmarkIndices] = LandmarkIndices(
        LEFT_EYE=np.array([36, 37, 38, 39, 40, 41], dtype=np.int32),
        RIGHT_EYE=np.array([42, 43, 44, 45, 46, 47], dtype=np.int32),
        LEFT_IRIS=np.array([], dtype=np.int32),
        RIGHT_IRIS=np.array([], dtype=np.int32),
        NOSE=30,
        HEAD_REFERENCE={"left_eye": 36, "right_eye": 45, "nose_tip": 30}
    )

    @classmethod
    def validate_indices(cls, config: LandmarkIndices) -> bool:
        all_indices = np.concatenate([
            config.LEFT_EYE,
            config.RIGHT_EYE,
            config.LEFT_IRIS,
            config.RIGHT_IRIS,
            np.array([config.NOSE], dtype=np.int32),
            np.array(list(config.HEAD_REFERENCE.values()), dtype=np.int32)
        ])
        if config is cls.MEDIAPIPE:
            return np.all(all_indices <= MAX_MEDIAPIPE_INDEX)
        elif config is cls.DLIB_COMPAT:
            return np.all(all_indices <= MAX_DLIB_INDEX)
        return False

DEFAULT: Final[LandmarkIndices] = FaceLandmarks.MEDIAPIPE

LEFT_EYE_EAR_IDX: Final[NDArray[np.int32]] = np.array([362, 385, 387, 263, 373, 380], dtype=np.int32)
RIGHT_EYE_EAR_IDX: Final[NDArray[np.int32]] = np.array([33, 160, 158, 133, 153, 144], dtype=np.int32)
