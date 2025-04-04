import os
import json

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        return json.load(f)

def save_metadata(metadata, output_path):
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

def filter_invalid_frames(metadata):
    return [entry for entry in metadata if entry["face_grid_data"]["IsValid"] == 1]

def augment_missing_data(metadata):
    for entry in metadata:
        if not entry.get("left_eye_images"):
            entry["left_eye_images"] = ["placeholder_left_eye.jpg"]
        if not entry.get("right_eye_images"):
            entry["right_eye_images"] = ["placeholder_right_eye.jpg"]
    return metadata

def normalize_gaze_coordinates(metadata):
    for entry in metadata:
        if "dot_data" in entry and "screen_data" in entry:
            XPts = entry["dot_data"].get("XPts")
            YPts = entry["dot_data"].get("YPts")
            W = entry["screen_data"].get("W")
            H = entry["screen_data"].get("H")
            if XPts is not None and YPts is not None and W and H:
                entry["dot_data"]["normalized_X"] = XPts / W
                entry["dot_data"]["normalized_Y"] = YPts / H
    return metadata

def process_metadata(input_path, output_path):
    metadata = load_metadata(input_path)
    metadata = filter_invalid_frames(metadata)
    metadata = augment_missing_data(metadata)
    metadata = normalize_gaze_coordinates(metadata)
    save_metadata(metadata, output_path)
    print(f"Processed metadata saved to {output_path}")

def process_all_folders(output_dir):
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if os.path.isdir(folder_path):
            input_metadata_path = os.path.join(folder_path, "metadata.json")
            output_metadata_path = os.path.join(folder_path, "processed_metadata.json")
            if os.path.exists(input_metadata_path):
                print(f"Processing metadata for folder: {folder}")
                process_metadata(input_metadata_path, output_metadata_path)
            else:
                print(f"Metadata file not found in folder: {folder}")

if __name__ == "__main__":
    output_directory = "output"
    process_all_folders(output_directory)
