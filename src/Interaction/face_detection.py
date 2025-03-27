# Interaction/face_detection.py
import cv2
from core.face_tracker import FaceMeshDetector
from core.utils import draw_landmarks
from typing import Any

class FaceDetection(FaceMeshDetector):
    def __init__(self) -> None:
        super().__init__()
    
    def process_frame(self, frame: cv2.UMat) -> cv2.UMat:
        landmarks_points = super().process_frame(frame)
        if landmarks_points is None:
            return frame
        # Draw all face landmarks on the frame.
        draw_landmarks(frame, landmarks_points)
        return frame

def main() -> None:
    detector = FaceDetection()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.process_frame(frame)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
