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
        if len(self.landmarks) >= 4:
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
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
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
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)
    
    def extract_eye_data(self, results, frame_shape):
        """
        Extract eye landmark data from face mesh results
        
        Args:
            results: MediaPipe FaceMesh results
            frame_shape: Shape of the input frame (height, width, channels)
            
        Returns:
            Tuple of (left_eye_data, right_eye_data) containing EyeData objects
        """
        if not results.multi_face_landmarks:
            return None, None
            
        height, width = frame_shape[:2]
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract left eye landmarks
        left_eye_landmarks = [(int(landmark.x * width), int(landmark.y * height), landmark.z) 
                             for idx, landmark in enumerate(face_landmarks.landmark) 
                             if idx in LEFT_EYE_INDICES]
        
        # Extract right eye landmarks
        right_eye_landmarks = [(int(landmark.x * width), int(landmark.y * height), landmark.z) 
                              for idx, landmark in enumerate(face_landmarks.landmark) 
                              if idx in RIGHT_EYE_INDICES]
        
        # Extract iris landmarks if available
        left_iris_landmarks = [(int(landmark.x * width), int(landmark.y * height), landmark.z) 
                              for idx, landmark in enumerate(face_landmarks.landmark) 
                              if idx in LEFT_IRIS_INDICES]
        
        right_iris_landmarks = [(int(landmark.x * width), int(landmark.y * height), landmark.z) 
                               for idx, landmark in enumerate(face_landmarks.landmark) 
                               if idx in RIGHT_IRIS_INDICES]
        
        left_eye_data = EyeData(left_eye_landmarks, left_iris_landmarks)
        right_eye_data = EyeData(right_eye_landmarks, right_iris_landmarks)
        
        return left_eye_data, right_eye_data
    
    def draw_landmarks(self, frame, results):
        """
        Draw facial landmarks on the frame
        
        Args:
            frame: BGR frame to draw on
            results: MediaPipe FaceMesh results
            
        Returns:
            Frame with landmarks drawn
        """
        if not results.multi_face_landmarks:
            return frame
            
        annotated_frame = frame.copy()
        
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
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
    
    def close(self):
        """Release resources"""
        self.face_mesh.close()

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
    
    # Calculate relative iris position within eye socket
    left_rel_x = (left_eye_data.iris_center[0] - left_eye_data.center[0]) 
    left_rel_y = (left_eye_data.iris_center[1] - left_eye_data.center[1])
    
    right_rel_x = (right_eye_data.iris_center[0] - right_eye_data.center[0])
    right_rel_y = (right_eye_data.iris_center[1] - right_eye_data.center[1])
    
    # Average the two eye directions
    gaze_x = (left_rel_x + right_rel_x) / 2
    gaze_y = (left_rel_y + right_rel_y) / 2
    
    return gaze_x, gaze_y
