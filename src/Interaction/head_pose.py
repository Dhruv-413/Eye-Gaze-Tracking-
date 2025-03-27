# Interaction/head_pose.py
import cv2
import numpy as np
from core.face_tracker import FaceMeshDetector
from Interaction.constants import NOSE_IDX, HEAD_LEFT_EYE_IDX, HEAD_RIGHT_EYE_IDX
from typing import Any

class HeadPoseEstimator(FaceMeshDetector):
    def __init__(self) -> None:
        super().__init__()
        self.last_landmarks = None
        self.eye_line_history = []  # Store recent eye_line values
        self.nose_to_eye_line_history = []  # Store recent nose_to_eye_line values
        self.smoothing_window = 5  # Number of frames for smoothing

    def smooth_value(self, value_list: list, new_value: float) -> float:
        """
        Smooth the value using a moving average.

        Args:
            value_list (list): List of recent values.
            new_value (float): New value to add.

        Returns:
            float: Smoothed value.
        """
        value_list.append(new_value)
        if len(value_list) > self.smoothing_window:
            value_list.pop(0)  # Maintain the window size
        return sum(value_list) / len(value_list)

    def estimate_pose(self, landmarks_points: np.ndarray) -> tuple:
        """
        Estimate a simplified head pose based on selected facial landmarks.

        Args:
            landmarks_points (np.ndarray): Array of facial landmark points.

        Returns:
            tuple: (eye_line, nose_to_eye_distance)
        """
        nose = landmarks_points[NOSE_IDX]
        left_eye = landmarks_points[HEAD_LEFT_EYE_IDX]
        right_eye = landmarks_points[HEAD_RIGHT_EYE_IDX]

        # Calculate distances
        eye_line = np.linalg.norm(left_eye - right_eye)
        nose_to_eye_line = np.linalg.norm(nose - (left_eye + right_eye) / 2)
        return eye_line, nose_to_eye_line
    
    def get_head_pose(self) -> str:
        """
        Returns a summary of the head pose information.
        """
        # Initialize default values
        eye_line = 0.0
        nose_to_eye_line = 0.0

        # If landmarks are available, calculate the pose
        if self.last_landmarks is not None:
            try:
                eye_line, nose_to_eye_line = self.estimate_pose(self.last_landmarks)
                print(f"Calculated Eye Line: {eye_line}, Nose Dist: {nose_to_eye_line}")
            except Exception as e:
                print(f"Error calculating head pose: {e}")

        return f"Eye Line: {eye_line:.2f}, Nose Dist: {nose_to_eye_line:.2f}"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        landmarks_points = super().process_frame(frame)
        if landmarks_points is None:
            print("No landmarks detected.")
            self.last_landmarks = None  # Ensure last_landmarks is reset
            return frame

        # Save the landmarks for use in get_head_pose
        self.last_landmarks = landmarks_points
        print(f"Landmarks updated: {self.last_landmarks}")

        eye_line, nose_to_eye_line = self.estimate_pose(landmarks_points)

        # Smooth the values
        smoothed_eye_line = self.smooth_value(self.eye_line_history, eye_line)
        smoothed_nose_to_eye_line = self.smooth_value(self.nose_to_eye_line_history, nose_to_eye_line)

        # Position these metrics in the bottom-right corner, just above the feedback text.
        height, width = frame.shape[:2]
        offset_x = 200
        offset_y = 80  # vertical offset so they don't overlap with EAR & blink info
        cv2.putText(
            frame,
            f"Eye Line: {smoothed_eye_line:.2f}",
            (width - offset_x, height - offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Nose Dist: {smoothed_nose_to_eye_line:.2f}",
            (width - offset_x, height - offset_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        return frame

def main() -> None:
    estimator = HeadPoseEstimator()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = estimator.process_frame(frame)
        # Remove cv2.imshow from here
        # cv2.imshow("Head Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
