# Interaction/pupil_detection.py
import cv2
import numpy as np
from core.face_tracker import FaceMeshDetector
from core.utils import draw_landmarks
from Interaction.eye_pupil_extract import EyeRegionExtractor
from typing import Optional, Tuple

class PupilDetector(FaceMeshDetector):
    def __init__(self) -> None:
        super().__init__()
        self.eye_extractor = EyeRegionExtractor()

    def detect_pupil(self, eye_region: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray]]:
        """
        Detect the pupil in the given eye region.

        Args:
            eye_region: A BGR image of the eye region.

        Returns:
            A tuple (pupil_center, contour) if a pupil is detected; otherwise, (None, None).
        """
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_eye = clahe.apply(gray_eye)
        blurred_eye = cv2.GaussianBlur(enhanced_eye, (5, 5), 0)
        thresholded_eye = cv2.adaptiveThreshold(
            blurred_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 < circularity <= 1.2:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return (cx, cy), contour
        return None, None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        landmarks_points = super().process_frame(frame)
        if landmarks_points is None:
            return frame

        left_eye, right_eye = self.eye_extractor.extract_eye_regions(frame)

        for eye_region, eye_name in zip([left_eye, right_eye], ["Left Eye", "Right Eye"]):
            if eye_region is None:
                continue
            pupil, contour = self.detect_pupil(eye_region)
            if pupil:
                cx, cy = pupil
                cv2.circle(eye_region, (cx, cy), 5, (0, 255, 0), -1)
                cv2.drawContours(eye_region, [contour], -1, (255, 0, 0), 1)
            cv2.imshow(eye_name, eye_region)

        return frame

def main() -> None:
    detector = PupilDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.process_frame(frame)
        cv2.imshow("Pupil Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
