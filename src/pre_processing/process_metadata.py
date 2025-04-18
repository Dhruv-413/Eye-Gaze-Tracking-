import os
import json

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        return json.load(f)

def save_metadata(metadata, metadata_path):
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def filter_invalid_entries(metadata):
    """Filter out entries with invalid or missing critical data"""
    filtered_metadata = []
    
    for entry in metadata:
        # Skip entries with missing critical data
        if not entry.get("face_grid_data") or entry.get("face_grid_data", {}).get("IsValid") != 1:
            continue
            
        # Keep entries that have all required data
        filtered_metadata.append(entry)
    
    return filtered_metadata

def normalize_coordinates(metadata):
    """Normalize coordinates in the metadata"""
    for entry in metadata:
        # Normalize dot data coordinates
        if "dot_data" in entry and "screen_data" in entry:
            XPts = entry["dot_data"].get("XPts")
            YPts = entry["dot_data"].get("YPts")
            W = entry["screen_data"].get("W")
            H = entry["screen_data"].get("H")
            
            if XPts is not None and YPts is not None and W and H:
                entry["dot_data"]["normalized_X"] = XPts / W
                entry["dot_data"]["normalized_Y"] = YPts / H
        
        # Normalize eyes midpoint coordinates
        if "eyes_midpoint" in entry and entry["eyes_midpoint"] and "image_dimensions" in entry:
            eyes_midpoint = entry["eyes_midpoint"]
            img_dims = entry["image_dimensions"]
            
            if "X" in eyes_midpoint and "Y" in eyes_midpoint and "width" in img_dims and "height" in img_dims:
                eyes_midpoint["normalized_X"] = eyes_midpoint["X"] / img_dims["width"]
                eyes_midpoint["normalized_Y"] = eyes_midpoint["Y"] / img_dims["height"]
    
    return metadata

def process_metadata(metadata_path):
    """Process metadata file: filter invalid entries, normalize coordinates, and update the original file"""
    print(f"Processing metadata file: {metadata_path}")
    
    # Load metadata
    metadata = load_metadata(metadata_path)
    original_count = len(metadata)
    
    # Process metadata
    metadata = filter_invalid_entries(metadata)
    metadata = normalize_coordinates(metadata)
    
    # Save back to original file
    save_metadata(metadata, metadata_path)
    
    filtered_count = len(metadata)
    print(f"Processed {metadata_path}: {filtered_count} valid entries from {original_count} total entries")

def process_all_folders(output_dir):
    """Process metadata.json in all subdirectories"""
    processed_count = 0
    
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if os.path.isdir(folder_path):
            metadata_path = os.path.join(folder_path, "metadata.json")
            if os.path.exists(metadata_path):
                process_metadata(metadata_path)
                processed_count += 1
            else:
                print(f"Metadata file not found in folder: {folder}")
    
    print(f"Processed metadata in {processed_count} folders")

if __name__ == "__main__":
    output_directory = "output"
    process_all_folders(output_directory)
