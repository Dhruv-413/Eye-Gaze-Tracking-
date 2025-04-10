import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array  #type: ignore[import]
import cv2
import glob

class GazeDataset:
    def __init__(self, root_dir, img_size=(224, 224), validation_split=0.2):
        """
        Initialize the eye gaze dataset
        
        Args:
            root_dir: Root directory containing the dataset subfolders
            img_size: Target image size for inputs (height, width)
            validation_split: Fraction of data to use for validation
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.validation_split = validation_split
        self.metadata_folders = self._get_metadata_folders()
        
    def _get_metadata_folders(self):
        """Find all folders with processed_metadata.json files"""
        folders = []
        for folder in glob.glob(os.path.join(self.root_dir, "output", "*")):
            metadata_path = os.path.join(folder, "processed_metadata.json")
            if os.path.exists(metadata_path):
                folders.append(folder)
        return folders
        
    def load_metadata(self, folder):
        """Load metadata from a specific folder"""
        metadata_path = os.path.join(folder, "processed_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image"""
        if not os.path.exists(image_path) or image_path.endswith("placeholder_left_eye.jpg") or image_path.endswith("placeholder_right_eye.jpg"):
            # Return zeros if image doesn't exist or is a placeholder
            return np.zeros((*self.img_size, 3))
        
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return np.zeros((*self.img_size, 3))
    
    def extract_metadata_features(self, metadata_entry):
        """Extract relevant features from metadata entry"""
        features = []
        
        # Add screen info
        if "screen_data" in metadata_entry:
            features.append(metadata_entry["screen_data"]["W"] / 1000.0)  # Normalize width
            features.append(metadata_entry["screen_data"]["H"] / 1000.0)  # Normalize height
            features.append(metadata_entry["screen_data"]["Orientation"] / 4.0)  # Normalize orientation
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # Add face grid data if available
        if "face_grid_data" in metadata_entry and isinstance(metadata_entry["face_grid_data"], dict):
            if "X" in metadata_entry["face_grid_data"]:
                features.append(metadata_entry["face_grid_data"]["X"] / 25.0)  # Normalize
            else:
                features.append(0.0)
                
            if "Y" in metadata_entry["face_grid_data"]:
                features.append(metadata_entry["face_grid_data"]["Y"] / 25.0)  # Normalize
            else:
                features.append(0.0)
                
            if "W" in metadata_entry["face_grid_data"]:
                features.append(metadata_entry["face_grid_data"]["W"] / 25.0)  # Normalize
            else:
                features.append(0.0)
                
            if "H" in metadata_entry["face_grid_data"]:
                features.append(metadata_entry["face_grid_data"]["H"] / 25.0)  # Normalize
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
            
        # Add some motion data if available
        if "motion_data" in metadata_entry and isinstance(metadata_entry["motion_data"], dict):
            if "Time" in metadata_entry["motion_data"]:
                features.append(metadata_entry["motion_data"]["Time"])  # Already normalized
            else:
                features.append(0.0)
            
            # Add DotNum as a feature
            if "DotNum" in metadata_entry["motion_data"]:
                features.append(metadata_entry["motion_data"]["DotNum"] / 23.0)  # Normalize (max is 23)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
            
        return np.array(features, dtype=np.float32)
    
    def create_dataset(self):
        """Create TensorFlow datasets for training and validation"""
        data = []
        
        for folder in self.metadata_folders:
            metadata = self.load_metadata(folder)
            
            for entry in metadata:
                # Skip entries without necessary information
                if not all(k in entry for k in ["face_image", "left_eye_image", "right_eye_image", "dot_data"]):
                    continue
                    
                if entry["left_eye_image"] is None or entry["right_eye_image"] is None:
                    continue
                    
                if "normalized_X" not in entry["dot_data"] or "normalized_Y" not in entry["dot_data"]:
                    continue
                
                # Get absolute paths for images
                face_path = os.path.join(self.root_dir, entry["face_image"])
                left_eye_path = os.path.join(self.root_dir, entry["left_eye_image"])
                right_eye_path = os.path.join(self.root_dir, entry["right_eye_image"])
                
                # Extract target gaze coordinates (normalized)
                gaze_target = np.array([
                    entry["dot_data"]["normalized_X"],
                    entry["dot_data"]["normalized_Y"]
                ], dtype=np.float32)
                
                # Extract metadata features
                metadata_features = self.extract_metadata_features(entry)
                
                data.append({
                    'face_path': face_path,
                    'left_eye_path': left_eye_path,
                    'right_eye_path': right_eye_path,
                    'metadata': metadata_features,
                    'gaze_target': gaze_target
                })
        
        # Shuffle and split the data
        np.random.shuffle(data)
        split_idx = int(len(data) * (1 - self.validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        return train_data, val_data
    
    def data_generator(self, data, batch_size=32):
        """Generator function to yield batches of data"""
        num_samples = len(data)
        indices = np.arange(num_samples)
        
        while True:
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Initialize batch arrays
                batch_left_eyes = np.zeros((len(batch_indices), *self.img_size, 3))
                batch_right_eyes = np.zeros((len(batch_indices), *self.img_size, 3))
                batch_faces = np.zeros((len(batch_indices), *self.img_size, 3))
                batch_metadata = np.zeros((len(batch_indices), len(data[0]['metadata'])))
                batch_targets = np.zeros((len(batch_indices), 2))
                
                # Fill the batch
                for i, idx in enumerate(batch_indices):
                    sample = data[idx]
                    
                    # Load and preprocess images
                    batch_left_eyes[i] = self.preprocess_image(sample['left_eye_path'])
                    batch_right_eyes[i] = self.preprocess_image(sample['right_eye_path'])
                    batch_faces[i] = self.preprocess_image(sample['face_path'])
                    
                    # Add metadata and target
                    batch_metadata[i] = sample['metadata']
                    batch_targets[i] = sample['gaze_target']
                
                # Yield the batch
                yield (
                    {
                        'left_eye_input': batch_left_eyes,
                        'right_eye_input': batch_right_eyes,
                        'face_input': batch_faces,
                        'metadata_input': batch_metadata
                    },
                    batch_targets
                )
