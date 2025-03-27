# Interaction/eye_pupil_extract.py
import cv2
import numpy as np
from core.face_tracker import FaceMeshDetector
from Interaction.constants import LEFT_EYE_PUPIL_IDX, RIGHT_EYE_PUPIL_IDX
from typing import Tuple, Optional

class EyeRegionExtractor(FaceMeshDetector):
    def __init__(self) -> None:
        super().__init__()
    
    def extract_eye_regions(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract the left and right eye regions from the frame.

        Args:
            frame (np.ndarray): BGR image.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (left_eye_region, right_eye_region)
        """
        landmarks_points = super().process_frame(frame)
        if landmarks_points is None:
            return None, None

        h, w, _ = frame.shape
        left_eye_coords = np.array([(int(pt[0]), int(pt[1])) for pt in landmarks_points[LEFT_EYE_PUPIL_IDX]])
        right_eye_coords = np.array([(int(pt[0]), int(pt[1])) for pt in landmarks_points[RIGHT_EYE_PUPIL_IDX]])

        left_eye_bbox = cv2.boundingRect(left_eye_coords)
        right_eye_bbox = cv2.boundingRect(right_eye_coords)

        left_eye_bbox = (
            max(0, left_eye_bbox[0]),
            max(0, left_eye_bbox[1]),
            min(left_eye_bbox[2], w - left_eye_bbox[0]),
            min(left_eye_bbox[3], h - left_eye_bbox[1])
        )
        right_eye_bbox = (
            max(0, right_eye_bbox[0]),
            max(0, right_eye_bbox[1]),
            min(right_eye_bbox[2], w - right_eye_bbox[0]),
            min(right_eye_bbox[3], h - right_eye_bbox[1])
        )

        left_eye_region = frame[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3],
                                left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]]
        right_eye_region = frame[right_eye_bbox[1]:right_eye_bbox[1] + right_eye_bbox[3],
                                 right_eye_bbox[0]:right_eye_bbox[0] + right_eye_bbox[2]]

        left_eye_region = cv2.GaussianBlur(left_eye_region, (3, 3), 0)
        right_eye_region = cv2.GaussianBlur(right_eye_region, (3, 3), 0)

        return left_eye_region, right_eye_region

def main() -> None:
    extractor = EyeRegionExtractor()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        left_eye, right_eye = extractor.extract_eye_regions(frame)
        if left_eye is not None and right_eye is not None:
            cv2.imshow("Left Eye", left_eye)
            cv2.imshow("Right Eye", right_eye)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
