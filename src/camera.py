import cv2
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Starting script...")

# Module imports with error logging.
try:
    from core.face_tracker import FaceMeshDetector
    logger.info("FaceMeshDetector imported successfully")
except Exception as e:
    logger.error("Error importing FaceMeshDetector: %s", e)
    
try:
    from Interaction.eye_blink import EyeBlinkDetector
    logger.info("EyeBlinkDetector imported successfully")
except Exception as e:
    logger.error("Error importing EyeBlinkDetector: %s", e)
    
try:
    from Interaction.eye_pupil_extract import EyeRegionExtractor
    logger.info("EyeRegionExtractor imported successfully")
except Exception as e:
    logger.error("Error importing EyeRegionExtractor: %s", e)
    
try:
    from Interaction.face_detection import FaceDetection
    logger.info("FaceDetection imported successfully")
except Exception as e:
    logger.error("Error importing FaceDetection: %s", e)
    
try:
    from Interaction.feedback import FeedbackSystem
    logger.info("FeedbackSystem imported successfully")
except Exception as e:
    logger.error("Error importing FeedbackSystem: %s", e)
    
try:
    from Interaction.head_pose import HeadPoseEstimator
    logger.info("HeadPoseEstimator imported successfully")
except Exception as e:
    logger.error("Error importing HeadPoseEstimator: %s", e)
    
try:
    from Interaction.pupil_detection import PupilDetector
    logger.info("PupilDetector imported successfully")
except Exception as e:
    logger.error("Error importing PupilDetector: %s", e)
    
try:
    from Interaction.constants import DEFAULT
    logger.info("DEFAULT imported successfully")
except Exception as e:
    logger.error("Error importing DEFAULT: %s", e)

def main():
    logger.info("Entering main function")
    
    # Initialize the face detector.
    try:
        logger.info("Initializing FaceMeshDetector...")
        face_mesh_detector = FaceMeshDetector()
        logger.info("FaceMeshDetector initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize FaceMeshDetector: %s", e)
        return

    # Initialize video capture.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        return
    logger.info("Camera opened successfully")

    # Initialize modules with dependencies.
    try:
        logger.info("Initializing modules...")
        # Pass DEFAULT configuration explicitly where required.
        eye_extractor = EyeRegionExtractor(face_mesh_detector, config=DEFAULT)
        blink_detector = EyeBlinkDetector(face_mesh_detector)
        face_detection = FaceDetection(face_mesh_detector)
        feedback_system = FeedbackSystem(face_mesh_detector, blink_detector)
        head_pose_estimator = HeadPoseEstimator(face_mesh_detector)
        pupil_detector = PupilDetector(face_mesh_detector, eye_extractor, landmarks_config=DEFAULT)
        logger.info("All modules initialized successfully")
    except Exception as e:
        logger.error("Module initialization failed: %s", e)
        cap.release()
        return

    # Create a resizable window.
    cv2.namedWindow("Integrated System", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                break

            # Resize frame early for consistent processing.
            frame = cv2.resize(frame, (1280, 720))
            logger.debug("Frame resized to 1280x720")

            try:
                # Run face detection.
                face_result = face_mesh_detector.process_frame(frame)
                if face_result is None:
                    logger.info("No face detected")
                    continue

                # Run additional processing modules.
                blink_result = blink_detector.process_frame(frame)
                pose_result = head_pose_estimator.process_frame(frame)
                pupil_result = pupil_detector.process_frame(frame)

                # Draw face landmarks.
                if face_result is not None:
                    face_mesh_detector.draw_landmarks(frame, face_result)

                # Extract and display cropped eye regions.
                left_eye, right_eye = eye_extractor.extract_eye_regions(frame)
                if left_eye.is_valid:
                    cv2.imshow("Left Eye", left_eye.image)
                if right_eye.is_valid:
                    cv2.imshow("Right Eye", right_eye.image)

                # Generate feedback text.
                feedback_text = feedback_system.get_feedback(
                    blink_result,
                    None,  # Gaze result removed
                    pose_result
                )

                # Prepare overlay text.
                overlay_text = [
                    f"Blinks: {blink_result.get('blink_count', 0)}",
                    f"Head Pose: {pose_result.euler_angles}" if pose_result else "Head Pose: Not detected",
                    feedback_text
                ]

                y_start = 30
                for text in overlay_text:
                    cv2.putText(
                        frame, text, (10, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    y_start += 30

                # Display the annotated frame.
                cv2.imshow("Integrated System", frame)

            except Exception as e:
                logger.error("Frame processing error: %s", e)

            # Exit on 'q' key press.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit requested")
                break

    finally:
        # Cleanup resources.
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released")

if __name__ == "__main__":
    main()
