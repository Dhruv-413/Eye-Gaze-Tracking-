# feedback.py
import cv2
import numpy as np
from core.face_tracker import FaceMeshDetector
from core.utils import calculate_ear, draw_landmarks
from Interaction.constants import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX
from typing import Any

class Feedback(FaceMeshDetector):
    def __init__(self, ear_threshold: float = 0.25, consec_frames: int = 3) -> None:
        super().__init__()
        self.EAR_THRESHOLD = ear_threshold
        self.CONSEC_FRAMES = consec_frames
        self.blink_count = 0
        self.frame_counter = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        landmarks_points = super().process_frame(frame)
        if landmarks_points is None:
            return frame

        # Draw the facial landmarks if desired:
        draw_landmarks(frame, landmarks_points)

        # Calculate EAR from the left and right eyes:
        left_eye = landmarks_points[LEFT_EYE_EAR_IDX]
        right_eye = landmarks_points[RIGHT_EYE_EAR_IDX]
        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

        # Simple blink detection logic
        if ear < self.EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.CONSEC_FRAMES:
                self.blink_count += 1
            self.frame_counter = 0

        # Place the text in the bottom-right corner
        height, width = frame.shape[:2]
        cv2.putText(
            frame, f"EAR: {ear:.2f}",
            (width - 200, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        cv2.putText(
            frame, f"Blinks: {self.blink_count}",
            (width - 200, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        return frame

    def get_feedback(self) -> str:
        """
        Returns feedback information, such as the blink count.
        """
        return f"Blinks: {self.blink_count}"

def main() -> None:
    feedback = Feedback()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = feedback.process_frame(frame)
        # Remove cv2.imshow from here
        # cv2.imshow("Feedback", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
