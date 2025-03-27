print("Starting script...")  # Debug log to confirm script starts

import cv2
print("cv2 imported successfully.")  # Debug log to confirm import

try:
    from Interaction.eye_blink import EyeBlinkDetector
    print("EyeBlinkDetector imported successfully.")
except Exception as e:
    print(f"Error importing EyeBlinkDetector: {e}")

try:
    from Interaction.face_detection import FaceDetection
    print("FaceDetection imported successfully.")
except Exception as e:
    print(f"Error importing FaceDetection: {e}")

try:
    from Interaction.feedback import Feedback
    print("Feedback imported successfully.")
except Exception as e:
    print(f"Error importing Feedback: {e}")

try:
    from Interaction.gaze_tracking import GazeTracker
    print("GazeTracker imported successfully.")
except Exception as e:
    print(f"Error importing GazeTracker: {e}")

try:
    from Interaction.head_pose import HeadPoseEstimator
    print("HeadPoseEstimator imported successfully.")
except Exception as e:
    print(f"Error importing HeadPoseEstimator: {e}")

try:
    from Interaction.pupil_detection import PupilDetector
    print("PupilDetector imported successfully.")
except Exception as e:
    print(f"Error importing PupilDetector: {e}")

def main():
    print("Entering main function...")  # Debug log to confirm main function starts

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully.")

    # Explicitly create a named window
    cv2.namedWindow("Integrated Interaction System", cv2.WINDOW_NORMAL)

    # Instantiate each module.
    try:
        print("Initializing modules...")
        blink_detector = EyeBlinkDetector()
        print("EyeBlinkDetector initialized.")
        face_detector = FaceDetection()
        print("FaceDetection initialized.")
        feedback_detector = Feedback()
        print("Feedback initialized.")
        gaze_tracker = GazeTracker()
        print("GazeTracker initialized.")
        head_pose_estimator = HeadPoseEstimator()
        print("HeadPoseEstimator initialized.")
        pupil_detector = PupilDetector()
        print("PupilDetector initialized.")
        print("All modules initialized successfully.")
    except Exception as e:
        print(f"Error initializing modules: {e}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        print("Frame captured successfully.")  # Debug log to confirm frame capture

        try:
            # Process frame with each module and add debug logs
            print("Processing frame with FaceDetection...")
            frame = face_detector.process_frame(frame)
            print("FaceDetection completed.")

            print("Processing frame with EyeBlinkDetector...")
            frame = blink_detector.process_frame(frame)
            print("EyeBlinkDetector completed.")

            print("Getting feedback...")
            feedback_text = feedback_detector.get_feedback()  # Call the actual method
            print(f"Feedback: {feedback_text}")

            print("Getting head pose...")
            head_pose_text = head_pose_estimator.get_head_pose()  # Assuming this method exists
            print(f"Head Pose: {head_pose_text}")

            print("Processing frame with GazeTracker...")
            try:
                frame = gaze_tracker.process_frame(frame)
                print("GazeTracker completed.")
            except Exception as e:
                print(f"Error in GazeTracker: {e}")

            print("Processing frame with PupilDetector...")
            frame = pupil_detector.process_frame(frame)
            print("PupilDetector completed.")

            # Resize the frame to increase dimensions
            frame = cv2.resize(frame, (1280, 720))  # Resize to 1280x720 resolution
            print("Frame resized to 1280x720.")

            # Debug log to confirm frame dimensions
            print(f"Frame dimensions: {frame.shape}")

            # Overlay feedback and head pose information on the frame.
            overlay_text = f"Feedback: {feedback_text}\nHead Pose: {head_pose_text}"
            y0, dy = frame.shape[0] - 50, 20  # Start near the bottom of the frame
            for i, line in enumerate(overlay_text.split('\n')):
                y = y0 + i * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Integrated Interaction System", frame)
            print("Frame displayed in OpenCV window.")  # Debug log to confirm display
        except Exception as e:
            print(f"Error processing frame: {e}")

        # Add a small delay to ensure the window updates
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting loop.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and all windows closed.")  # Debug log for cleanup

if __name__ == "__main__":
    main()
