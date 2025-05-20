import cv2
import numpy as np
import mediapipe as mp
import dataclasses
from typing import List, Tuple, Dict, Optional

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Define constants for eye landmarks
# Indices based on MediaPipe Face Mesh topology
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]


@dataclasses.dataclass
class EyeData:
    """Class for storing eye landmark data and metrics"""
    landmarks: List[Tuple[float, float, float]]
    iris_landmarks: List[Tuple[float, float, float]]
    center: Optional[Tuple[float, float, float]] = None
    iris_center: Optional[Tuple[float, float, float]] = None
    aspect_ratio: float = 0.0
    
    def __post_init__(self):
        if self.landmarks:
            x_coords = [p[0] for p in self.landmarks]
            y_coords = [p[1] for p in self.landmarks]
            z_coords = [p[2] for p in self.landmarks]
            self.center = (
                sum(x_coords) / len(self.landmarks),
                sum(y_coords) / len(self.landmarks),
                sum(z_coords) / len(self.landmarks)
            )
            
        if self.iris_landmarks:
            x_coords = [p[0] for p in self.iris_landmarks]
            y_coords = [p[1] for p in self.iris_landmarks]
            z_coords = [p[2] for p in self.iris_landmarks]
            self.iris_center = (
                sum(x_coords) / len(self.iris_landmarks),
                sum(y_coords) / len(self.iris_landmarks),
                sum(z_coords) / len(self.iris_landmarks)
            )
        
        # Calculate eye aspect ratio if we have enough landmarks
        if len(self.landmarks) >= 6:
            # Simplified EAR calculation
            height = (abs(self.landmarks[1][1] - self.landmarks[5][1]) + 
                      abs(self.landmarks[2][1] - self.landmarks[4][1])) / 2
            width = abs(self.landmarks[0][0] - self.landmarks[3][0])
            self.aspect_ratio = height / width if width > 0 else 0


class FaceMeshDetector:
    def __init__(self, 
                 static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize the face mesh detector with MediaPipe
        
        Args:
            static_image_mode: Whether to treat input as static image vs video stream
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum detection confidence threshold
            min_tracking_confidence: Minimum tracking confidence threshold
        """
        try:
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            print("MediaPipe FaceMesh initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe FaceMesh: {e}")
            raise
    
    def process_frame(self, frame):
        """
        Process a frame and detect face landmarks
        
        Args:
            frame: RGB frame to process
            
        Returns:
            Results from MediaPipe FaceMesh processing
        """
        # MediaPipe requires RGB input
        if frame is None:
            return None
            
        try:
            # Extract dimensions for MediaPipe
            h, w = frame.shape[:2]
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add explicit image dimensions to address the NORM_RECT warning
            results = self.face_mesh.process(rgb_frame)
            
            # Check if we have valid results
            if not results or not hasattr(results, 'multi_face_landmarks'):
                return None
                
            return results
        except Exception as e:
            print(f"Error processing frame with MediaPipe: {e}")
            return None
    
    def extract_eye_data(self, results, frame_shape):
        """
        Extract eye landmark data from face mesh results
        
        Args:
            results: MediaPipe FaceMesh results
            frame_shape: Shape of the input frame (height, width, channels)
            
        Returns:
            Tuple of (left_eye_data, right_eye_data) containing EyeData objects
        """
        if not results or not hasattr(results, 'multi_face_landmarks') or not results.multi_face_landmarks:
            return None, None
            
        height, width = frame_shape[:2]
        
        try:
            # Get the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract left eye landmarks
            left_eye_landmarks = []
            for idx in LEFT_EYE_INDICES:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    left_eye_landmarks.append((
                        int(landmark.x * width), 
                        int(landmark.y * height), 
                        landmark.z
                    ))
            
            # Extract right eye landmarks
            right_eye_landmarks = []
            for idx in RIGHT_EYE_INDICES:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    right_eye_landmarks.append((
                        int(landmark.x * width), 
                        int(landmark.y * height), 
                        landmark.z
                    ))
            
            # Extract iris landmarks if available
            left_iris_landmarks = []
            for idx in LEFT_IRIS_INDICES:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    left_iris_landmarks.append((
                        int(landmark.x * width), 
                        int(landmark.y * height), 
                        landmark.z
                    ))
            
            right_iris_landmarks = []
            for idx in RIGHT_IRIS_INDICES:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    right_iris_landmarks.append((
                        int(landmark.x * width), 
                        int(landmark.y * height), 
                        landmark.z
                    ))
            
            left_eye_data = EyeData(left_eye_landmarks, left_iris_landmarks)
            right_eye_data = EyeData(right_eye_landmarks, right_iris_landmarks)
            
            return left_eye_data, right_eye_data
            
        except Exception as e:
            print(f"Error extracting eye data: {e}")
            return None, None
    
    def draw_landmarks(self, frame, results):
        """
        Draw facial landmarks on the frame
        
        Args:
            frame: BGR frame to draw on
            results: MediaPipe FaceMesh results or individual face_landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        try:
            annotated_frame = frame.copy()
            
            # Handle both full results object and individual face_landmarks
            if hasattr(results, 'multi_face_landmarks'):
                face_landmarks_list = results.multi_face_landmarks
            else:
                # Assume results is a single face_landmarks
                face_landmarks_list = [results]
            
            if not face_landmarks_list:
                return frame
                
            for face_landmarks in face_landmarks_list:
                # Draw face mesh with reduced opacity
                connections = mp_face_mesh.FACEMESH_TESSELATION
                connection_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=connection_spec
                )
                
                # Draw eye contours
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Draw iris contours if refine_landmarks was enabled
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )
                
            return annotated_frame
        except Exception as e:
            print(f"Error drawing landmarks: {e}")
            return frame
    
    def close(self):
        """Release resources"""
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh:
                self.face_mesh.close()
                print("MediaPipe FaceMesh closed")
        except Exception as e:
            print(f"Error closing FaceMesh: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def estimate_gaze_direction(left_eye_data, right_eye_data):
    """
    Estimate gaze direction based on eye and iris positions
    
    Args:
        left_eye_data: EyeData for left eye
        right_eye_data: EyeData for right eye
    
    Returns:
        (x, y) tuple representing the relative gaze direction
    """
    if not left_eye_data or not right_eye_data:
        return None
    
    if not left_eye_data.iris_center or not right_eye_data.iris_center:
        return None
    
    try:
        # Calculate relative iris position within eye socket
        left_rel_x = (left_eye_data.iris_center[0] - left_eye_data.center[0]) 
        left_rel_y = (left_eye_data.iris_center[1] - left_eye_data.center[1])
        
        right_rel_x = (right_eye_data.iris_center[0] - right_eye_data.center[0])
        right_rel_y = (right_eye_data.iris_center[1] - right_eye_data.center[1])
        
        # Average the two eye directions
        gaze_x = (left_rel_x + right_rel_x) / 2
        gaze_y = (left_rel_y + right_rel_y) / 2
        
        return gaze_x, gaze_y
    except Exception as e:
        print(f"Error estimating gaze direction: {e}")
        return None
