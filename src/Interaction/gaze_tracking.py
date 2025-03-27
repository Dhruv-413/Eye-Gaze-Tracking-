# Interaction/gaze_tracking.py
import cv2
import numpy as np
from core.face_tracker import FaceMeshDetector
from Interaction.eye_pupil_extract import EyeRegionExtractor
from typing import Any

class GazeTracker(FaceMeshDetector):
    def __init__(self, threshold_value: int = 70) -> None:
        super().__init__()
        self.threshold_value = threshold_value
        self.eye_extractor = EyeRegionExtractor()
    
    def calculate_gaze_ratio(self, eye_region: np.ndarray) -> float:
        """
        Calculate the horizontal gaze ratio for the given eye region.

        Args:
            eye_region (np.ndarray): Cropped grayscale image of the eye.

        Returns:
            float: Horizontal gaze ratio.
        """
        # Convert to binary image (thresholding)
        _, threshold_eye = cv2.threshold(eye_region, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Divide the eye into left and right halves
        height, width = threshold_eye.shape
        left_half = threshold_eye[:, :width // 2]
        right_half = threshold_eye[:, width // 2:]

        # Count white pixels in each half
        left_white = cv2.countNonZero(left_half)
        right_white = cv2.countNonZero(right_half)

        # Avoid division by zero
        if left_white == 0:
            left_white = 1
        if right_white == 0:
            right_white = 1

        # Calculate gaze ratio
        gaze_ratio = right_white / left_white
        return gaze_ratio
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        landmarks_points = super().process_frame(frame)
        if landmarks_points is None:
            print("No landmarks detected.")
            return frame

        # Adjust unpacking logic to handle unexpected return values
        eye_regions = self.eye_extractor.extract_eye_regions(frame)
        if isinstance(eye_regions, tuple) and len(eye_regions) == 2:
            left_eye, right_eye = eye_regions
            if left_eye is not None and right_eye is not None:
                left_gaze_ratio = self.calculate_gaze_ratio(left_eye)
                right_gaze_ratio = self.calculate_gaze_ratio(right_eye)
                gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2.0
                print(f"Left Gaze Ratio: {left_gaze_ratio}, Right Gaze Ratio: {right_gaze_ratio}, Average: {gaze_ratio}")
                cv2.putText(frame, f"Gaze Ratio: {gaze_ratio:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("Eye regions not detected.")
        else:
            print("Unexpected return value from extract_eye_regions.")
        return frame

def main() -> None:
    tracker = GazeTracker()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = tracker.process_frame(frame)
        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
