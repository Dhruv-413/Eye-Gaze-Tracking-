import os
import json
from PIL import Image
from PIL import ImageOps  # Import ImageOps for grayscale conversion
from PIL import ImageEnhance  # Import ImageEnhance for brightness, contrast, and color adjustments
import cv2  # Import OpenCV
import numpy as np  # Import NumPy

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
    motion_data = load_json("motion.json")
    screen_data = load_json("screen.json")

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

        # Function to perform moderate geometric and photometric augmentation
        def augment_and_save(image, base_path, prefix, is_eye_region=False):
            paths = []
            # Save original image
            original_path = f"{base_path}_{prefix}.jpg"
            image.save(original_path)
            paths.append(original_path)

            # Horizontal flip (adjust labels accordingly if needed)
            if not is_eye_region:  # Avoid flipping eye regions
                flipped_path = f"{base_path}_{prefix}_flipped.jpg"
                flipped = ImageOps.mirror(image)
                flipped.save(flipped_path)
                paths.append(flipped_path)

            # Slight rotation (small angles)
            for angle in [-10, 10]:  # Rotate by -10 and +10 degrees
                rotated_path = f"{base_path}_{prefix}_rotated_{angle}.jpg"
                rotated = image.rotate(angle, expand=True)
                rotated.save(rotated_path)
                paths.append(rotated_path)

            # Slight cropping and scaling
            if not is_eye_region:  # Avoid aggressive cropping/scaling on eye regions
                width, height = image.size
                crop_margin = 0.1  # Crop 10% from each side
                cropped = image.crop((
                    int(crop_margin * width),
                    int(crop_margin * height),
                    int((1 - crop_margin) * width),
                    int((1 - crop_margin) * height)
                ))
                cropped = cropped.resize((width, height))  # Scale back to original size
                cropped_path = f"{base_path}_{prefix}_cropped.jpg"
                cropped.save(cropped_path)
                paths.append(cropped_path)

            # Photometric augmentations (brightness only for eye regions)
            for factor in [0.9, 1.1]:  # Slightly decrease/increase brightness
                brightness_path = f"{base_path}_{prefix}_brightness_{factor}.jpg"
                brightness = ImageEnhance.Brightness(image).enhance(factor)
                brightness.save(brightness_path)
                paths.append(brightness_path)

                if not is_eye_region:  # Additional photometric changes for non-eye regions
                    # Contrast adjustment
                    contrast_path = f"{base_path}_{prefix}_contrast_{factor}.jpg"
                    contrast = ImageEnhance.Contrast(image).enhance(factor)
                    contrast.save(contrast_path)
                    paths.append(contrast_path)

                    # Color adjustment
                    color_path = f"{base_path}_{prefix}_color_{factor}.jpg"
                    color = ImageEnhance.Color(image).enhance(factor)
                    color.save(color_path)
                    paths.append(color_path)

            return paths

        # Extract face bounding box
        face_files = None
        if i < len(apple_face["IsValid"]) and apple_face["IsValid"][i]:
            face_crop = frame.crop((
                apple_face["X"][i],
                apple_face["Y"][i],
                apple_face["X"][i] + apple_face["W"][i],
                apple_face["Y"][i] + apple_face["H"][i]
            ))
            face_file_base = os.path.join(face_dir, f"frame_{i}_face")
            face_files = augment_and_save(face_crop, face_file_base, "face")

        # Extract left eye bounding box
        left_eye_files = None
        if i < len(apple_left_eye["IsValid"]) and apple_left_eye["IsValid"][i] and face_files:
            left_eye_crop = face_crop.crop((
                apple_left_eye["X"][i],
                apple_left_eye["Y"][i],
                apple_left_eye["X"][i] + apple_left_eye["W"][i],
                apple_left_eye["Y"][i] + apple_left_eye["H"][i]
            ))
            left_eye_file_base = os.path.join(left_eye_dir, f"frame_{i}_left_eye")
            left_eye_files = augment_and_save(left_eye_crop, left_eye_file_base, "left_eye", is_eye_region=True)

        # Extract right eye bounding box
        right_eye_files = None
        if i < len(apple_right_eye["IsValid"]) and apple_right_eye["IsValid"][i] and face_files:
            right_eye_crop = face_crop.crop((
                apple_right_eye["X"][i],
                apple_right_eye["Y"][i],
                apple_right_eye["X"][i] + apple_right_eye["W"][i],
                apple_right_eye["Y"][i] + apple_right_eye["H"][i]
            ))
            right_eye_file_base = os.path.join(right_eye_dir, f"frame_{i}_right_eye")
            right_eye_files = augment_and_save(right_eye_crop, right_eye_file_base, "right_eye", is_eye_region=True)

        # Link with dot information
        dot_data = {
            "DotNum": dot_info["DotNum"][i] if i < len(dot_info["DotNum"]) else None,
            "XPts": dot_info["XPts"][i] if i < len(dot_info["XPts"]) else None,
            "YPts": dot_info["YPts"][i] if i < len(dot_info["YPts"]) else None,
            "XCam": dot_info["XCam"][i] if i < len(dot_info["XCam"]) else None,
            "YCam": dot_info["YCam"][i] if i < len(dot_info["YCam"]) else None,
            "Time": dot_info["Time"][i] if i < len(dot_info["Time"]) else None
        }

        # Link with face grid information
        face_grid_data = {
            "X": face_grid["X"][i] if i < len(face_grid["X"]) else None,
            "Y": face_grid["Y"][i] if i < len(face_grid["Y"]) else None,
            "W": face_grid["W"][i] if i < len(face_grid["W"]) else None,
            "H": face_grid["H"][i] if i < len(face_grid["H"]) else None,
            "IsValid": face_grid["IsValid"][i] if i < len(face_grid["IsValid"]) else None
        }

        # Link with motion data (if available)
        motion_entry = motion_data[i] if motion_data and i < len(motion_data) else None

        # Link with screen data
        screen_entry = {
            "H": screen_data["H"][i] if i < len(screen_data["H"]) else None,
            "W": screen_data["W"][i] if i < len(screen_data["W"]) else None,
            "Orientation": screen_data["Orientation"][i] if i < len(screen_data["Orientation"]) else None
        }

        # Append metadata with augmented image paths
        metadata.append({
            "frame": frame_file,
            "face_images": face_files,
            "left_eye_images": left_eye_files,
            "right_eye_images": right_eye_files,
            "dot_data": dot_data,
            "face_grid_data": face_grid_data,
            "motion_data": motion_entry,
            "screen_data": screen_entry
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
parent_directory = "c:\\Users\\dhruv\\OneDrive\\Desktop\\minor project\\Extraction\\gazecapture.part\\gazecapture"
output_directory = "c:\\Users\\dhruv\\OneDrive\\Desktop\\minor project\\Extraction\\output"
process_all_folders(parent_directory, output_directory)