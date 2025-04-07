"""
Module for loading and processing gaze tracking data from the metadata files.
"""

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional

class GazeDataLoader:
    """Class to load and process gaze tracking data from metadata files."""
    
    def __init__(self, metadata_path: str, img_size: Tuple[int, int] = (64, 64), batch_size: int = 1000):
        """
        Initialize the data loader with the path to metadata.
        
        Args:
            metadata_path: Path to the processed_metadata.json file
            img_size: Target size for face images (width, height)
            batch_size: Number of samples to process at once to save memory
        """
        self.metadata_path = metadata_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.base_dir = os.path.dirname(metadata_path)
        
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for the model.
        
        Args:
            img: Input image in BGR format (as loaded by OpenCV)
            
        Returns:
            Preprocessed image
        """
        # Resize the image
        img_resized = cv2.resize(img, self.img_size)
        
        # Convert to RGB if it's a color image
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        return img_resized
    
    def load_data(self, include_eyes: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Load all valid samples from the metadata file.
        
        Args:
            include_eyes: Whether to include left and right eye images
            
        Returns:
            Tuple of (image_dict, gaze_points) where image_dict contains 'faces', 'left_eyes', 'right_eyes'
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as file:
            data = json.load(file)
        
        total_samples = len(data)
        print(f"Found {total_samples} total samples in metadata")
        
        # Initialize arrays for different image types
        face_images = []
        left_eye_images = []
        right_eye_images = []
        gaze_points = []
        processed = 0
        skipped = 0
        missing_files = 0
        
        # Check for the image directories
        faces_dir = os.path.join(self.base_dir, 'faces')
        left_eyes_dir = os.path.join(self.base_dir, 'left_eyes')
        right_eyes_dir = os.path.join(self.base_dir, 'right_eyes')
        
        # Check if directories exist
        dirs_exist = {
            'faces': os.path.exists(faces_dir),
            'left_eyes': os.path.exists(left_eyes_dir),
            'right_eyes': os.path.exists(right_eyes_dir)
        }
        
        for dir_name, exists in dirs_exist.items():
            if exists:
                dir_path = os.path.join(self.base_dir, dir_name)
                print(f"Found {dir_name} directory: {dir_path}")
                print(f"{dir_name} image count: {len(os.listdir(dir_path))}")
                if os.listdir(dir_path):
                    print(f"Sample {dir_name} images: {', '.join(sorted(os.listdir(dir_path))[:3])}")
        
        # Index files by frame number for quick lookups
        file_indices = {}
        
        for dir_name, dir_path in [
            ('faces', faces_dir), 
            ('left_eyes', left_eyes_dir), 
            ('right_eyes', right_eyes_dir)
        ]:
            if dirs_exist[dir_name]:
                file_indices[dir_name] = {}
                for filename in os.listdir(dir_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        # Extract frame number from filename
                        import re
                        match = re.search(r'frame_(\d+)', filename)
                        if match:
                            frame_id = match.group(1)
                            file_indices[dir_name][frame_id] = os.path.join(dir_path, filename)
                
                print(f"Indexed {len(file_indices[dir_name])} {dir_name} images")
        
        # Process frames in batches to save memory
        for idx in range(0, total_samples, self.batch_size):
            batch_data = data[idx:min(idx+self.batch_size, total_samples)]
            batch_face_images = []
            batch_left_eye_images = []
            batch_right_eye_images = []
            batch_gaze = []
            
            for frame_idx, frame in enumerate(batch_data):
                try:
                    # Check for required fields
                    if 'dot_data' not in frame:
                        skipped += 1
                        continue
                        
                    # Check if normalized gaze coordinates exist
                    if ('normalized_X' not in frame['dot_data'] or 
                        'normalized_Y' not in frame['dot_data']):
                        skipped += 1
                        continue
                    
                    # Extract gaze coordinates
                    gaze_x = frame['dot_data']['normalized_X']
                    gaze_y = frame['dot_data']['normalized_Y']
                    
                    # Skip non-numeric or out-of-range values
                    if not (isinstance(gaze_x, (int, float)) and isinstance(gaze_y, (int, float))):
                        skipped += 1
                        continue
                    
                    # Skip extreme values (likely errors)
                    if not (0 <= gaze_x <= 1.2) or not (0 <= gaze_y <= 1.2):
                        skipped += 1
                        continue
                    
                    # Find frame number
                    frame_num = None
                    
                    # Try to get frame ID from metadata fields
                    if 'face_image' in frame:
                        match = re.search(r'frame_(\d+)', frame['face_image'])
                        if match:
                            frame_num = match.group(1)
                    elif 'face_images' in frame and frame['face_images']:
                        match = re.search(r'frame_(\d+)', frame['face_images'][0])
                        if match:
                            frame_num = match.group(1)
                    elif 'frame_path' in frame:
                        match = re.search(r'frame_(\d+)', frame['frame_path'])
                        if match:
                            frame_num = match.group(1)
                    elif 'id' in frame:
                        frame_num = str(frame['id'])
                    
                    # Debug first few or every 500th frame
                    if idx + frame_idx < 3 or (idx + frame_idx) % 500 == 0:
                        print(f"Frame {idx+frame_idx}: Looking for frame number {frame_num}")
                    
                    # Check if we have all required images
                    have_face = frame_num in file_indices.get('faces', {})
                    have_left_eye = frame_num in file_indices.get('left_eyes', {})
                    have_right_eye = frame_num in file_indices.get('right_eyes', {})
                    
                    # Only include sample if we have face, or if we want eye images, both eyes
                    if have_face and ((not include_eyes) or (have_left_eye and have_right_eye)):
                        # Load face
                        face_path = file_indices['faces'][frame_num]
                        face_img = cv2.imread(face_path)
                        
                        # If we're using eye images, load them too
                        if include_eyes:
                            left_eye_path = file_indices['left_eyes'][frame_num]
                            right_eye_path = file_indices['right_eyes'][frame_num]
                            
                            left_eye_img = cv2.imread(left_eye_path)
                            right_eye_img = cv2.imread(right_eye_path)
                            
                            # Skip if any image couldn't be loaded
                            if face_img is None or left_eye_img is None or right_eye_img is None:
                                missing_files += 1
                                continue
                            
                            # Preprocess all images
                            face_img = self.preprocess_image(face_img)
                            left_eye_img = self.preprocess_image(left_eye_img)
                            right_eye_img = self.preprocess_image(right_eye_img)
                            
                            batch_face_images.append(face_img)
                            batch_left_eye_images.append(left_eye_img)
                            batch_right_eye_images.append(right_eye_img)
                        else:
                            # Skip if face couldn't be loaded
                            if face_img is None:
                                missing_files += 1
                                continue
                                
                            # Preprocess face image
                            face_img = self.preprocess_image(face_img)
                            batch_face_images.append(face_img)
                        
                        batch_gaze.append([gaze_x, gaze_y])
                    else:
                        missing_files += 1
                except Exception as e:
                    print(f"Error processing sample {idx+frame_idx}: {e}")
                    skipped += 1
            
            if batch_face_images:
                face_images.extend(batch_face_images)
                if include_eyes:
                    left_eye_images.extend(batch_left_eye_images)
                    right_eye_images.extend(batch_right_eye_images)
                gaze_points.extend(batch_gaze)
                
            processed += len(batch_data)
            print(f"Processed {processed}/{total_samples} samples, " 
                  f"loaded {len(face_images)} valid samples, "
                  f"skipped {skipped}, missing files {missing_files}")
            
        if not face_images:
            print("\nInspecting dataset directory structure...")
            if os.path.exists(self.base_dir):
                print(f"Base directory exists: {self.base_dir}")
                for root, dirs, files in os.walk(self.base_dir):
                    rel_path = os.path.relpath(root, self.base_dir)
                    if rel_path == '.' or rel_path.count(os.sep) == 0:
                        print(f"Directory: {root}")
                        if dirs:
                            print(f"  Subdirectories: {dirs}")
                        if files and len(files) < 10:
                            print(f"  Files: {files}")
                        else:
                            print(f"  File count: {len(files)}")
            
            raise ValueError("No valid images found in the dataset! Check the paths in your metadata file.")
            
        print(f"Successfully loaded {len(face_images)} valid samples")
        
        # Convert to numpy arrays
        result_dict = {
            'faces': np.array(face_images, dtype=np.float32) / 255.0
        }
        
        if include_eyes and left_eye_images and right_eye_images:
            result_dict['left_eyes'] = np.array(left_eye_images, dtype=np.float32) / 255.0
            result_dict['right_eyes'] = np.array(right_eye_images, dtype=np.float32) / 255.0
            print(f"Included {len(left_eye_images)} left eye and {len(right_eye_images)} right eye images")
        
        gaze_points = np.array(gaze_points, dtype=np.float32)
        
        return result_dict, gaze_points
    
    @staticmethod
    def find_available_datasets(base_dir: str) -> List[str]:
        """
        Find all available dataset directories with processed_metadata.json files
        
        Args:
            base_dir: Base directory to search for datasets
            
        Returns:
            List of paths to metadata files
        """
        available_datasets = []
        if os.path.exists(base_dir):
            for folder in os.listdir(base_dir):
                dataset_path = os.path.join(base_dir, folder)
                if not os.path.isdir(dataset_path):
                    continue
                    
                # Check for processed metadata file first (preferred)
                processed_metadata = os.path.join(dataset_path, 'processed_metadata.json')
                if os.path.exists(processed_metadata):
                    available_datasets.append(processed_metadata)
                    continue
                
                # Fall back to regular metadata if processed not available
                metadata = os.path.join(dataset_path, 'metadata.json')
                if os.path.exists(metadata):
                    available_datasets.append(metadata)
                    
        # Print a summary of found datasets
        if available_datasets:
            print(f"Found {len(available_datasets)} dataset(s):")
            for i, path in enumerate(available_datasets, 1):
                folder = os.path.basename(os.path.dirname(path))
                print(f"  {i}. {folder} - {os.path.basename(path)}")
        else:
            print(f"No datasets found in {base_dir}. Checking directory structure...")
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    item_path = os.path.join(base_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  Directory: {item}")
                        metadata_candidates = [f for f in os.listdir(item_path) if f.endswith('.json')]
                        if metadata_candidates:
                            print(f"    JSON files: {metadata_candidates}")
        
        return available_datasets

    def extract_head_pose(self, indices=None) -> np.ndarray:
        """
        Extract head pose data (pitch, yaw, roll) from metadata.
        
        Args:
            indices: Optional indices to extract data for
            
        Returns:
            np.ndarray with shape (N, 3) containing pitch, yaw, roll values
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as file:
            data = json.load(file)
            
        # Use specified indices or all data
        if indices is not None:
            samples = [data[i] for i in indices if i < len(data)]
        else:
            samples = data
            
        head_pose = []
        for sample in samples:
            # Default values
            pitch, yaw, roll = 0.0, 0.0, 0.0
            
            # Try to extract from motion_data
            if 'motion_data' in sample and isinstance(sample['motion_data'], dict):
                motion = sample['motion_data']
                if 'AttitudePitch' in motion and isinstance(motion['AttitudePitch'], (int, float)):
                    pitch = float(motion['AttitudePitch'])
                if 'AttitudeYaw' in motion and isinstance(motion['AttitudeYaw'], (int, float)):
                    yaw = float(motion['AttitudeYaw'])
                if 'AttitudeRoll' in motion and isinstance(motion['AttitudeRoll'], (int, float)):
                    roll = float(motion['AttitudeRoll'])
                    
            head_pose.append([pitch, yaw, roll])
            
        return np.array(head_pose, dtype=np.float32)
        
    def extract_metadata_features(self, indices=None, feature_count=6) -> np.ndarray:
        """
        Extract metadata features useful for gaze prediction.
        
        Args:
            indices: Optional indices to extract data for
            feature_count: Number of features to extract
            
        Returns:
            np.ndarray with shape (N, feature_count)
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as file:
            data = json.load(file)
            
        # Use specified indices or all data
        if indices is not None:
            samples = [data[i] for i in indices if i < len(data)]
        else:
            samples = data
            
        metadata = []
        for sample in samples:
            features = np.zeros(feature_count, dtype=np.float32)
            
            # Extract screen orientation and dimensions
            if 'screen_data' in sample and isinstance(sample['screen_data'], dict):
                screen = sample['screen_data']
                if 'W' in screen and 'H' in screen:
                    # Normalize dimensions
                    features[0] = float(screen.get('W', 0)) / 1000.0
                    features[1] = float(screen.get('H', 0)) / 1000.0
                    # Aspect ratio
                    if features[1] > 0:
                        features[2] = features[0] / features[1]
                # Screen orientation
                if 'Orientation' in screen:
                    features[3] = float(screen.get('Orientation', 1)) / 4.0
            
            # Face grid data for face position on screen
            if 'face_grid_data' in sample and isinstance(sample['face_grid_data'], dict):
                face_grid = sample['face_grid_data']
                if all(k in face_grid for k in ['X', 'Y', 'W', 'H']):
                    # Normalized position and size
                    features[4] = float(face_grid.get('X', 0) + face_grid.get('W', 0)/2) / 25.0
                    features[5] = float(face_grid.get('Y', 0) + face_grid.get('H', 0)/2) / 25.0
            
            metadata.append(features)
            
        return np.array(metadata, dtype=np.float32)
