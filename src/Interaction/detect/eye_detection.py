import cv2
import logging
import time
import numpy as np
from core.face import FaceMeshDetector, FaceDetectionResult
from core.eye import crop_eye_region
from core.config import LEFT_EYE_IDX, RIGHT_EYE_IDX
from utils.camera_utils import open_camera

# Configure logging with both file and stream handlers.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eye_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_eye_detection(camera_id: int = 0, resolution: tuple = (1280, 720), 
                      display_scale: float = 2.0) -> bool:
    """
    Run the eye detection application. It captures frames from the specified camera,
    processes each frame to extract facial landmarks, crops the left and right eye regions
    based on configured landmark indices, and displays the results.
    
    Args:
        camera_id (int): The ID of the camera device.
        resolution (tuple): Desired resolution (width, height) for the frames.
        display_scale (float): Scale factor for displaying the eye region.
    
    Returns:
        bool: True if the application ran successfully, False otherwise.
    """
    logger.info(f"Starting eye detection using camera {camera_id} with resolution {resolution}.")

    # Performance metrics
    frame_count = 0
    start_time = time.time()
    total_start_time = time.time()
    fps = 0.0

    # Detection analytics
    face_detected_frames = 0
    total_frames = 0

    # Define window names and positions
    main_window_name = "Eye Detection"
    left_eye_window = "Left Eye"
    right_eye_window = "Right Eye"

    # Create windows and initialize blank eye image for no detection case.
    cv2.namedWindow(main_window_name)
    cv2.namedWindow(left_eye_window)
    cv2.namedWindow(right_eye_window)
    cv2.moveWindow(main_window_name, 50, 50)
    cv2.moveWindow(left_eye_window, 50, resolution[1] + 80)
    cv2.moveWindow(right_eye_window, 350, resolution[1] + 80)
    
    # Create a blank eye image
    blank_eye = np.zeros((100, 150, 3), dtype=np.uint8)
    cv2.putText(blank_eye, "No face detected", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    try:
        # Initialize face detector (for landmark extraction)
        with FaceMeshDetector() as detector:
            # Initialize camera capture using utility function
            cap = open_camera(camera_id, resolution)
            if cap is None:
                logger.error(f"Error: Could not open camera (device ID: {camera_id}).")
                return False

            # Display initial blank eye images
            cv2.imshow(left_eye_window, blank_eye)
            cv2.imshow(right_eye_window, blank_eye)

            # Main processing loop
            while True:
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame. Reopening camera...")
                    cap.release()
                    cap = open_camera(camera_id, resolution)
                    if cap is None:
                        logger.error("Failed to reopen camera.")
                        break
                    continue

                # Resize frame to desired resolution
                frame = cv2.resize(frame, resolution)
                total_frames += 1

                # Extract facial landmarks
                detection_result: FaceDetectionResult = detector.process_frame(frame)
                frame_processing_time = time.time() - frame_start_time

                # Update FPS metrics
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                    logger.debug(f"Current FPS: {fps:.2f}")

                # Create an overlay for performance metrics
                info_overlay = np.zeros((150, resolution[0], 3), dtype=np.uint8)
                cv2.putText(info_overlay, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_overlay, f"Processing time: {frame_processing_time*1000:.1f}ms", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if detection_result is not None:
                    face_detected_frames += 1
                    face_detection_rate = (face_detected_frames / total_frames) * 100
                    cv2.putText(info_overlay, f"Face detection rate: {face_detection_rate:.1f}%", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(info_overlay, f"Runtime: {int(time.time()-total_start_time)}s", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Crop left and right eye regions using configured indices
                    left_eye_crop = crop_eye_region(frame, detection_result.landmarks, LEFT_EYE_IDX)
                    right_eye_crop = crop_eye_region(frame, detection_result.landmarks, RIGHT_EYE_IDX)

                    # Display cropped eye regions, scaling up if needed
                    if left_eye_crop is not None:
                        h_eye, w_eye = left_eye_crop.shape[:2]
                        left_eye_display = cv2.resize(left_eye_crop, (int(w_eye * display_scale), int(h_eye * display_scale)))
                        cv2.imshow(left_eye_window, left_eye_display)
                    else:
                        cv2.imshow(left_eye_window, blank_eye)

                    if right_eye_crop is not None:
                        h_eye, w_eye = right_eye_crop.shape[:2]
                        right_eye_display = cv2.resize(right_eye_crop, (int(w_eye * display_scale), int(h_eye * display_scale)))
                        cv2.imshow(right_eye_window, right_eye_display)
                    else:
                        cv2.imshow(right_eye_window, blank_eye)

                    # Draw face landmarks and bounding box on the main frame
                    detector.draw_landmarks(frame, detection_result.landmarks)
                    detector.draw_face_bounding_box(frame, detection_result)
                    
                    # Optionally, mark eye landmark points on the main frame
                    for idx in LEFT_EYE_IDX:
                        x, y = detection_result.landmarks[idx]
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    for idx in RIGHT_EYE_IDX:
                        x, y = detection_result.landmarks[idx]
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                else:
                    face_detection_rate = (face_detected_frames / total_frames) * 100 if total_frames > 0 else 0
                    cv2.putText(info_overlay, f"Face detection rate: {face_detection_rate:.1f}%", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(info_overlay, f"Runtime: {int(time.time()-total_start_time)}s", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "No face detected", (resolution[0]//3, resolution[1]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(left_eye_window, blank_eye)
                    cv2.imshow(right_eye_window, blank_eye)
                    
                # Combine main frame with info overlay for display
                combined_display = np.vstack((frame, info_overlay))
                cv2.imshow(main_window_name, combined_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exit requested by user.")
                    break
                elif key == ord('r'):
                    face_detected_frames = 0
                    total_frames = 0
                    total_start_time = time.time()
                    logger.info("Statistics reset by user.")
                    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return False
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        total_runtime = time.time() - total_start_time
        face_detection_rate = (face_detected_frames / total_frames) * 100 if total_frames > 0 else 0
        logger.info(f"Session summary - Runtime: {total_runtime:.1f}s, Frames: {total_frames}, Detection rate: {face_detection_rate:.1f}%")
        logger.info("Resources released successfully.")
    
    return True

if __name__ == "__main__":
    run_eye_detection()
