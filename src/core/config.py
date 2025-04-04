# constants.py
import numpy as np
from typing import Final, Dict, Optional
from dataclasses import dataclass, field
from numpy.typing import NDArray

# Maximum valid index for MediaPipe FaceMesh landmarks.
MAX_MEDIAPIPE_INDEX: Final[int] = 467

@dataclass(frozen=True)
class LandmarkIndices:
    """
    Immutable data class representing facial landmark indices.
    """
    LEFT_EYE: NDArray[np.int32]
    RIGHT_EYE: NDArray[np.int32]
    LEFT_IRIS: NDArray[np.int32]
    RIGHT_IRIS: NDArray[np.int32]
    NOSE: int
    HEAD_REFERENCE: Dict[str, int]

    def is_valid(self) -> bool:
        """
        Validate that all landmark indices do not exceed the maximum allowed index.
        """
        all_indices = np.concatenate([
            self.LEFT_EYE,
            self.RIGHT_EYE,
            self.LEFT_IRIS,
            self.RIGHT_IRIS,
            np.array([self.NOSE], dtype=np.int32),
            np.array(list(self.HEAD_REFERENCE.values()), dtype=np.int32)
        ])
        return np.all(all_indices <= MAX_MEDIAPIPE_INDEX)

@dataclass(frozen=True)
class FaceLandmarksConfig:
    """
    Configuration for facial landmark indices.
    Currently supports MediaPipe's 468-point model.
    """
    MEDIAPIPE: LandmarkIndices = field(default_factory=lambda: LandmarkIndices(
        LEFT_EYE=np.array([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], dtype=np.int32),
        RIGHT_EYE=np.array([263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466], dtype=np.int32),
        LEFT_IRIS=np.array([474, 475, 476, 477], dtype=np.int32),
        RIGHT_IRIS=np.array([469, 470, 471, 472], dtype=np.int32),
        NOSE=1,
        HEAD_REFERENCE={"left_eye": 33, "right_eye": 263, "nose_tip": 4}
    ))

    def validate(self) -> bool:
        """
        Validate the configuration of landmark indices.
        """
        return self.MEDIAPIPE.is_valid()

# Default landmark configuration instance.
DEFAULT: Final[LandmarkIndices] = FaceLandmarksConfig().MEDIAPIPE

# Default indices used for various computations:
LEFT_EYE_IDX: Final[NDArray[np.int32]] = DEFAULT.LEFT_EYE
RIGHT_EYE_IDX: Final[NDArray[np.int32]] = DEFAULT.RIGHT_EYE
LEFT_IRIS_IDX: Final[NDArray[np.int32]] = DEFAULT.LEFT_IRIS
RIGHT_IRIS_IDX: Final[NDArray[np.int32]] = DEFAULT.RIGHT_IRIS
NOSE_IDX: Final[int] = DEFAULT.NOSE
HEAD_REFERENCE_IDX: Final[Dict[str, int]] = DEFAULT.HEAD_REFERENCE

# Additional configuration parameters for eye analysis:
@dataclass(frozen=True)
class EyeAnalysisConfig:
    """
    Configuration for eye analysis tasks such as blink detection.
    """
    EAR_THRESHOLD: float = 0.25  # Typical threshold for eye closure.
    MIN_EYE_SIZE: int = 10       # Minimum pixel dimension for a valid eye region.

EYE_CONFIG: Final[EyeAnalysisConfig] = EyeAnalysisConfig()

# Configuration for iris detection:
@dataclass(frozen=True)
class IrisDetectionConfig:
    """
    Configuration parameters for iris detection.
    """
    MIN_IRIS_AREA: float = 100.0   # Empirical minimum iris area in pixels.
    MAX_IRIS_AREA: float = 1000.0  # Empirical maximum iris area in pixels.
    CONFIDENCE_THRESHOLD: float = 0.5  # Confidence threshold for valid iris detection.

IRIS_CONFIG: Final[IrisDetectionConfig] = IrisDetectionConfig()

@dataclass(frozen=True)
class PupilDetectionConfig:
    """
    Configuration parameters for pupil detection.
    """
    min_pupil_area: float = 20.0      # Minimum pupil area in pixels.
    max_pupil_area: float = 300.0     # Maximum pupil area in pixels.
    min_circularity: float = 0.7      # Minimum circularity for a valid pupil.
    adaptive_thresh_block: int = 11   # Block size for adaptive thresholding.
    adaptive_thresh_c: int = 2        # Constant subtracted from the mean in thresholding.
    clahe_clip_limit: float = 2.0     # Clip limit for CLAHE.
    blur_kernel_size: int = 5         # Kernel size for Gaussian blur.
    confidence_threshold: float = 0.4 # Threshold for accepting a pupil detection.

PUPIL_CONFIG: Final[PupilDetectionConfig] = PupilDetectionConfig()

@dataclass(frozen=True)
class EyeBlinkConfig:
    """
    Configuration for eye blink detection.
    """
    EAR_THRESHOLD: float = 0.25      # EAR below which the eye is considered closed.
    SMOOTHING_WINDOW: int = 5        # Number of frames for temporal smoothing.
    CONSEC_FRAMES: int = 3           # Minimum consecutive frames below threshold to register a blink.

EYE_BLINK_CONFIG: Final[EyeBlinkConfig] = EyeBlinkConfig()


@dataclass(frozen=True)
class HeadPoseConfig:
    """
    Configuration parameters for head pose estimation.
    
    method: "PNP", "WHENET", or "HYBRID" (to combine both approaches).
    model_points: 3D reference points for the PnP solution.
    camera_matrix: Intrinsic camera matrix (if None, will be computed dynamically).
    dist_coeffs: Distortion coefficients.
    whenet_model_path: Path to the pre-trained WHEnet model.
    weight_pnp: Weight for the PnP estimation in HYBRID mode.
    weight_whenet: Weight for the WHEnet estimation in HYBRID mode.
    history_size: Number of past frames to keep for temporal smoothing.
    """
    method: str = "HYBRID"  # Options: "PNP", "WHENET", "HYBRID"
    model_points: NDArray[np.float32] = np.array([
        [0.0, 0.0, 0.0],        # Nose tip
        [0.0, -0.3, -0.1],      # Chin
        [-0.15, 0.15, -0.1],    # Left eye left corner
        [0.15, 0.15, -0.1],     # Right eye right corner
        [-0.1, -0.1, -0.1],     # Left mouth corner
        [0.1, -0.1, -0.1]       # Right mouth corner
    ], dtype=np.float32)
    camera_matrix: Optional[NDArray[np.float32]] = None
    dist_coeffs: Optional[NDArray[np.float32]] = None
    whenet_model_path: str = "path/to/whenet/model"
    weight_pnp: float = 0.7
    weight_whenet: float = 0.3
    history_size: int = 5

HEAD_POSE_CONFIG: Final[HeadPoseConfig] = HeadPoseConfig()