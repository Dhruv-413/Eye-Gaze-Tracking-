# face_detection.py
from typing import Tuple
import cv2
import logging
import time

from core.face import FaceMeshDetector, FaceDetectionResult
from core.config import DEFAULT  # DEFAULT is your landmark configuration
from utils.image_utils import draw_rectangle, draw_text
from utils.logging_utils import configure_logging

logger = configure_logging("face_detection.log")

def run_face_detection(camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)) -> bool:
    """
    Run the face detection application with the specified camera and resolution.
    
    Args:
        camera_id (int): Camera device ID.
        resolution (Tuple[int, int]): (Width, Height) for frame resizing.
    
    Returns:
        bool: True if the application ran successfully, False otherwise.
    """
    logger.info(f"Starting face detection application using camera {camera_id}.")

    # Initialize performance metrics
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        # Initialize the face detector with the default landmark configuration
        with FaceMeshDetector(landmark_config=DEFAULT) as detector:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Error: Could not open the webcam (device ID: {camera_id}).")
                return False

            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            logger.info(f"Camera initialized with resolution {resolution}.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame from webcam. Attempting to reconnect...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        logger.error("Failed to reconnect to camera.")
                        break
                    continue

                # Resize frame to the specified resolution
                frame = cv2.resize(frame, resolution)

                # Process the frame to get the face detection result
                result: FaceDetectionResult = detector.process_frame(frame)

                # Update performance metrics (FPS calculation)
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                    logger.debug(f"Current FPS: {fps:.2f}")

                # Overlay FPS on the frame
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if result is not None:
                    # Draw bounding box and confidence using utility functions
                    draw_rectangle(frame, result.bounding_box)
                    draw_text(frame, f"Confidence: {result.confidence:.2f}", (10, 30))
                else:
                    # Indicate when no face is detected
                    cv2.putText(frame, "No face detected", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the processed frame
                cv2.imshow("Face Detection", frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exit requested by user.")
                    break
                
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return False
    finally:
        # Ensure resources are released
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released successfully.")

    return True

if __name__ == "__main__":
    run_face_detection()
