import os
import json
from PIL import Image
import cv2
import numpy as np
import math

def extract_images_with_annotations(data_dir, output_dir):
    # List of required JSON files
    required_files = [
        "frames.json", "appleFace.json", "appleLeftEye.json", 
        "appleRightEye.json", "dotInfo.json", "faceGrid.json", 
        "motion.json", "screen.json"
    ]

    # Check for missing files
    missing_files = [file for file in required_files if not os.path.exists(os.path.join(data_dir, file))]
    if missing_files:
        print(f"Skipping folder {data_dir} due to missing files: {', '.join(missing_files)}")
        print(f"Ensure the folder structure is correct and the required files are present.")
        return

    # Load JSON files
    def load_json(file_name):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path) as f:
            return json.load(f)

    # Load required JSON files
    frames = load_json("frames.json")
    apple_face = load_json("appleFace.json")
    apple_left_eye = load_json("appleLeftEye.json")
    apple_right_eye = load_json("appleRightEye.json")
    dot_info = load_json("dotInfo.json")
    face_grid = load_json("faceGrid.json")
    screen_data = load_json("screen.json")
    
    # Load motion data if available (it may be empty in some datasets)
    try:
        motion_data = load_json("motion.json")
    except json.JSONDecodeError:
        print(f"Warning: motion.json in {data_dir} is empty or corrupt. Using empty motion data.")
        motion_data = []

    # Skip folder if critical files are missing
    if not frames or not apple_face or not dot_info:
        print(f"Skipping folder {data_dir} due to missing critical files.")
        return

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    face_dir = os.path.join(output_dir, "faces")
    left_eye_dir = os.path.join(output_dir, "left_eyes")
    right_eye_dir = os.path.join(output_dir, "right_eyes")
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(left_eye_dir, exist_ok=True)
    os.makedirs(right_eye_dir, exist_ok=True)

    metadata = []

    # Process each frame
    for i, frame_file in enumerate(frames):
        frame_path = os.path.join(data_dir, "frames", frame_file)
        if not os.path.exists(frame_path):
            continue

        # Open the frame image
        frame = Image.open(frame_path)

        # Convert the frame to a NumPy array for OpenCV processing
        frame_array = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Convert the frame to grayscale using OpenCV
        frame_gray = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)

        # Convert back to PIL Image for further processing
        frame = Image.fromarray(frame_gray)

        # Simple function to save image without augmentation
        def save_image(image, output_path):
            image.save(output_path)
            return output_path

        # Extract face bounding box
        face_file = None
        if i < len(apple_face["IsValid"]) and apple_face["IsValid"][i]:
            face_crop = frame.crop((
                apple_face["X"][i],
                apple_face["Y"][i],
                apple_face["X"][i] + apple_face["W"][i],
                apple_face["Y"][i] + apple_face["H"][i]
            ))
            face_file_path = os.path.join(face_dir, f"frame_{i}_face.jpg")
            face_file = save_image(face_crop, face_file_path)

        # Extract left eye bounding box (relative to face crop)
        left_eye_file = None
        left_eye_abs_coords = None
        if i < len(apple_left_eye["IsValid"]) and apple_left_eye["IsValid"][i] and face_file:
            # Calculate absolute coordinates in original image
            left_eye_abs_x = apple_face["X"][i] + apple_left_eye["X"][i]
            left_eye_abs_y = apple_face["Y"][i] + apple_left_eye["Y"][i]
            left_eye_abs_coords = {
                "X": left_eye_abs_x,
                "Y": left_eye_abs_y,
                "W": apple_left_eye["W"][i],
                "H": apple_left_eye["H"][i]
            }
            
            # Crop from face image
            left_eye_crop = face_crop.crop((
                apple_left_eye["X"][i],
                apple_left_eye["Y"][i],
                apple_left_eye["X"][i] + apple_left_eye["W"][i],
                apple_left_eye["Y"][i] + apple_left_eye["H"][i]
            ))
            left_eye_file_path = os.path.join(left_eye_dir, f"frame_{i}_left_eye.jpg")
            left_eye_file = save_image(left_eye_crop, left_eye_file_path)

        # Extract right eye bounding box (relative to face crop)
        right_eye_file = None
        right_eye_abs_coords = None
        if i < len(apple_right_eye["IsValid"]) and apple_right_eye["IsValid"][i] and face_file:
            # Calculate absolute coordinates in original image
            right_eye_abs_x = apple_face["X"][i] + apple_right_eye["X"][i]
            right_eye_abs_y = apple_face["Y"][i] + apple_right_eye["Y"][i]
            right_eye_abs_coords = {
                "X": right_eye_abs_x,
                "Y": right_eye_abs_y,
                "W": apple_right_eye["W"][i],
                "H": apple_right_eye["H"][i]
            }
            
            # Crop from face image
            right_eye_crop = face_crop.crop((
                apple_right_eye["X"][i],
                apple_right_eye["Y"][i],
                apple_right_eye["X"][i] + apple_right_eye["W"][i],
                apple_right_eye["Y"][i] + apple_right_eye["H"][i]
            ))
            right_eye_file_path = os.path.join(right_eye_dir, f"frame_{i}_right_eye.jpg")
            right_eye_file = save_image(right_eye_crop, right_eye_file_path)

        # Get screen dimensions for normalization
        screen_width = screen_data["W"][i] if i < len(screen_data["W"]) else None
        screen_height = screen_data["H"][i] if i < len(screen_data["H"]) else None
        screen_orientation = screen_data["Orientation"][i] if i < len(screen_data["Orientation"]) else None

        # Extract dot information with normalized coordinates
        dot_data = {}
        if i < len(dot_info["DotNum"]):
            dot_data = {
                "DotNum": dot_info["DotNum"][i],
                # Raw point coordinates (in points)
                "XPts": dot_info["XPts"][i],
                "YPts": dot_info["YPts"][i],
                # Camera-relative coordinates (in cm)
                "XCam": dot_info["XCam"][i],
                "YCam": dot_info["YCam"][i],
                "Time": dot_info["Time"][i]
            }
            
            # Add normalized screen coordinates (between 0 and 1)
            if screen_width is not None and screen_height is not None:
                dot_data["normalized_X"] = dot_info["XPts"][i] / screen_width
                dot_data["normalized_Y"] = dot_info["YPts"][i] / screen_height

        # Extract face grid information
        face_grid_data = {}
        if i < len(face_grid["IsValid"]):
            face_grid_data = {
                "X": face_grid["X"][i],
                "Y": face_grid["Y"][i],
                "W": face_grid["W"][i],
                "H": face_grid["H"][i],
                "IsValid": face_grid["IsValid"][i]
            }
            
            # Add normalized face grid coordinates (between 0 and 1)
            face_grid_data["normalized_X"] = face_grid["X"][i] / 25.0  # Grid is 25x25
            face_grid_data["normalized_Y"] = face_grid["Y"][i] / 25.0
            face_grid_data["normalized_W"] = face_grid["W"][i] / 25.0
            face_grid_data["normalized_H"] = face_grid["H"][i] / 25.0

        # Link with screen data
        screen_entry = {
            "H": screen_height,
            "W": screen_width,
            "Orientation": screen_orientation
        }

        # Extract motion data if available
        motion_entry = None
        head_pose = None
        
        # Attempt to extract head pose from motion data first
        if motion_data and i < len(motion_data) and motion_data[i]:
            motion_entry = motion_data[i]
            
            # Extract head pose from motion data (if available)
            if isinstance(motion_entry, dict):
                # Check for attitude data which contains orientation
                if 'attitude' in motion_entry:
                    head_pose = {
                        "pitch": motion_entry['attitude']['pitch'],
                        "yaw": motion_entry['attitude']['yaw'],
                        "roll": motion_entry['attitude']['roll']
                    }
                # Some datasets use quaternion components
                elif all(key in motion_entry for key in ['quaternionX', 'quaternionY', 'quaternionZ', 'quaternionW']):
                    qx, qy, qz, qw = [motion_entry[f'quaternion{axis}'] for axis in 'XYZW']
                    # Convert quaternion to Euler angles
                    pitch = math.asin(2.0 * (qw * qy - qz * qx))
                    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qz * qz))
                    head_pose = {
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll
                    }
                
                # Extract other motion data (accelerometer, gyroscope, magnetometer)
                for key in ['gravity', 'userAcceleration', 'rotationRate', 'magneticField']:
                    if key in motion_entry and isinstance(motion_entry[key], dict):
                        if motion_entry is None:
                            motion_entry = {}
                        motion_entry[f"{key}_processed"] = {
                            "x": motion_entry[key].get("x"),
                            "y": motion_entry[key].get("y"),
                            "z": motion_entry[key].get("z")
                        }
        
        # If head_pose is still None and we have face/eye detections, estimate head pose from facial landmarks
        if head_pose is None and left_eye_abs_coords and right_eye_abs_coords and face_file:
            # Calculate eye centers
            left_eye_center_x = left_eye_abs_coords["X"] + left_eye_abs_coords["W"] / 2
            left_eye_center_y = left_eye_abs_coords["Y"] + left_eye_abs_coords["H"] / 2
            right_eye_center_x = right_eye_abs_coords["X"] + right_eye_abs_coords["W"] / 2
            right_eye_center_y = right_eye_abs_coords["Y"] + right_eye_abs_coords["H"] / 2
            
            # Calculate eye distance (interpupillary distance)
            eye_distance = math.sqrt((right_eye_center_x - left_eye_center_x)**2 + 
                                    (right_eye_center_y - left_eye_center_y)**2)
            
            # Calculate the angle between eyes (roll)
            if right_eye_center_x != left_eye_center_x:
                roll = math.atan2(right_eye_center_y - left_eye_center_y, 
                                right_eye_center_x - left_eye_center_x)
            else:
                roll = 0
            
            # Get face dimensions
            face_width = apple_face["W"][i]
            face_height = apple_face["H"][i]
            
            # Estimate yaw from the horizontal position of face in the frame
            # (This is a simplified approximation)
            face_center_x = apple_face["X"][i] + face_width / 2
            yaw_estimate = ((face_center_x / frame.width) - 0.5) * math.pi/2
            
            # Estimate pitch from vertical position of eyes relative to face
            eyes_center_y = (left_eye_center_y + right_eye_center_y) / 2
            face_center_y = apple_face["Y"][i] + face_height / 2
            relative_eye_pos = (eyes_center_y - face_center_y) / (face_height / 2)
            pitch_estimate = -relative_eye_pos * math.pi/6  # Scaled approximation
            
            # Create head pose estimate
            head_pose = {
                "pitch": pitch_estimate,
                "yaw": yaw_estimate,
                "roll": roll,
                "eye_distance": eye_distance,
                "estimated_from_face": True  # Flag that this is an estimate, not from sensors
            }

        # Calculate mid-point between eyes (if both eyes were detected)
        eyes_midpoint = None
        if left_eye_abs_coords and right_eye_abs_coords:
            left_eye_center_x = left_eye_abs_coords["X"] + left_eye_abs_coords["W"] / 2
            left_eye_center_y = left_eye_abs_coords["Y"] + left_eye_abs_coords["H"] / 2
            right_eye_center_x = right_eye_abs_coords["X"] + right_eye_abs_coords["W"] / 2
            right_eye_center_y = right_eye_abs_coords["Y"] + right_eye_abs_coords["H"] / 2
            
            eyes_midpoint = {
                "X": (left_eye_center_x + right_eye_center_x) / 2,
                "Y": (left_eye_center_y + right_eye_center_y) / 2
            }
            
            # Normalize midpoint by image dimensions
            if frame.width > 0 and frame.height > 0:
                eyes_midpoint["normalized_X"] = eyes_midpoint["X"] / frame.width
                eyes_midpoint["normalized_Y"] = eyes_midpoint["Y"] / frame.height

        # Append metadata with original image paths only (no augmentation)
        metadata.append({
            "frame": frame_file,
            "frame_path": os.path.join(data_dir, "frames", frame_file),
            "face_image": face_file,
            "left_eye_image": left_eye_file,
            "right_eye_image": right_eye_file,
            "left_eye_coords": left_eye_abs_coords,
            "right_eye_coords": right_eye_abs_coords,
            "eyes_midpoint": eyes_midpoint,
            "dot_data": dot_data,
            "face_grid_data": face_grid_data,
            "motion_data": motion_entry,
            "head_pose": head_pose,
            "screen_data": screen_entry,
            "image_dimensions": {
                "width": frame.width,
                "height": frame.height
            }
        })

    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_file}")

def process_all_folders(parent_dir, output_dir):
    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
            extract_images_with_annotations(folder_path, os.path.join(output_dir, folder))
        else:
            print(f"Skipping non-directory item: {folder_path}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process eye gaze dataset')
    parser.add_argument('--parent_dir', type=str, default="dataset", 
                        help='Parent directory containing subject folders')
    parser.add_argument('--output_dir', type=str, default="output",
                        help='Output directory for extracted data')
    
    args = parser.parse_args()
    process_all_folders(args.parent_dir, args.output_dir)