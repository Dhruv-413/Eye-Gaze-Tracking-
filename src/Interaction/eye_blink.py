# Interaction/eye_blink.py
import cv2
import numpy as np
from core.face_tracker import FaceMeshDetector
from core.utils import calculate_ear
from Interaction.constants import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX
from typing import Any

class EyeBlinkDetector(FaceMeshDetector):
    def __init__(self, ear_threshold: float = 0.2, consec_frames: int = 3) -> None:
        super().__init__()
        self.EAR_THRESHOLD = ear_threshold
        self.CONSEC_FRAMES = consec_frames
        self.blink_count = 0
        self.frame_counter = 0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        landmarks_points = super().process_frame(frame)
        if landmarks_points is None:
            return frame

        left_eye = landmarks_points[LEFT_EYE_EAR_IDX]
        right_eye = landmarks_points[RIGHT_EYE_EAR_IDX]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.CONSEC_FRAMES:
                self.blink_count += 1
            self.frame_counter = 0

        cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.polylines(frame, [left_eye.astype(np.int32)], True, (255, 0, 0), 1)
        cv2.polylines(frame, [right_eye.astype(np.int32)], True, (255, 0, 0), 1)
        return frame

def main() -> None:
    detector = EyeBlinkDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.process_frame(frame)
        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
