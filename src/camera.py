print("Starting script...")

import cv2
print("cv2 imported successfully")

try:
    from core.face_tracker import FaceMeshDetector
    print("FaceMeshDetector imported successfully")
except Exception as e:
    print(f"Error importing FaceMeshDetector: {e}")

try:
    from Interaction.eye_blink import EyeBlinkDetector
    print("EyeBlinkDetector imported successfully")
except Exception as e:
    print(f"Error importing EyeBlinkDetector: {e}")

try:
    from Interaction.eye_pupil_extract import EyeRegionExtractor
    print("EyeRegionExtractor imported successfully")
except Exception as e:
    print(f"Error importing EyeRegionExtractor: {e}")

try:
    from Interaction.face_detection import FaceDetection
    print("FaceDetection imported successfully")
except Exception as e:
    print(f"Error importing FaceDetection: {e}")

try:
    from Interaction.feedback import FeedbackSystem
    print("FeedbackSystem imported successfully")
except Exception as e:
    print(f"Error importing FeedbackSystem: {e}")

try:
    from Interaction.head_pose import HeadPoseEstimator
    print("HeadPoseEstimator imported successfully")
except Exception as e:
    print(f"Error importing HeadPoseEstimator: {e}")

try:
    from Interaction.pupil_detection import PupilDetector
    print("PupilDetector imported successfully")
except Exception as e:
    print(f"Error importing PupilDetector: {e}")

try:
    from Interaction.constants import DEFAULT
    print("DEFAULT imported successfully")
except Exception as e:
    print(f"Error importing DEFAULT: {e}")

def main():
    print("Entering main function")
    
    # Initialize core detector first
    try:
        print("Initializing FaceMeshDetector...")
        face_mesh_detector = FaceMeshDetector()
        print("FaceMeshDetector initialized successfully")
    except Exception as e:
        print(f"Failed to initialize FaceMeshDetector: {e}")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    print("Camera opened successfully")

    # Set up modules with dependencies
    try:
        print("Initializing modules...")
        
        # Pass DEFAULT config explicitly where required
        eye_extractor = EyeRegionExtractor(face_mesh_detector, config=DEFAULT)
        blink_detector = EyeBlinkDetector(face_mesh_detector)
        face_detection = FaceDetection(face_mesh_detector)
        feedback_system = FeedbackSystem(face_mesh_detector, blink_detector)
        head_pose_estimator = HeadPoseEstimator(face_mesh_detector)
        pupil_detector = PupilDetector(face_mesh_detector, eye_extractor, landmarks_config=DEFAULT)
        
        print("All modules initialized successfully")
    except Exception as e:
        print(f"Module initialization failed: {e}")
        cap.release()
        return

    # Create window with normalized size
    cv2.namedWindow("Integrated System", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Resize early for consistent processing
            frame = cv2.resize(frame, (1280, 720))
            print("Frame resized to 1280x720")

            # Process frame through pipeline
            try:
                # Face detection
                face_result = face_mesh_detector.process_frame(frame)
                if face_result is None:
                    print("No face detected")
                    continue

                # Run processing pipeline
                blink_result = blink_detector.process_frame(frame)
                pose_result = head_pose_estimator.process_frame(frame)
                pupil_result = pupil_detector.process_frame(frame)

                # Draw landmarks
                if face_result is not None:
                    face_mesh_detector.draw_landmarks(frame, face_result)

                # Extract and display cropped eye regions
                left_eye, right_eye = eye_extractor.extract_eye_regions(frame)
                if left_eye.is_valid:
                    cv2.imshow("Left Eye", left_eye.image)
                if right_eye.is_valid:
                    cv2.imshow("Right Eye", right_eye.image)

                # Generate feedback
                feedback_text = feedback_system.get_feedback(
                    blink_result,
                    None,  # Removed gaze_result
                    pose_result
                )

                # Prepare overlay text
                overlay_text = [
                    f"Blinks: {blink_result['blink_count']}",
                    f"Head Pose: {pose_result.euler_angles}" if pose_result else "Head Pose: Not detected",
                    feedback_text
                ]

                # Draw overlay
                y_start = 30
                for text in overlay_text:
                    cv2.putText(frame, text, (10, y_start), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_start += 30

                # Show frame with landmarks
                cv2.imshow("Integrated System", frame)
                
            except Exception as e:
                print(f"Frame processing error: {e}")

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested")
                break

    finally:
        # Cleanup resources
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()