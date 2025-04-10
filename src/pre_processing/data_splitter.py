import os
import json
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def load_processed_metadata(metadata_path):
    """Load processed metadata from a JSON file."""
    with open(metadata_path, "r") as f:
        return json.load(f)

def create_dataset_directories(output_base_dir):
    """Create directory structure for train and test datasets."""
    train_dir = os.path.join(output_base_dir, "train")
    test_dir = os.path.join(output_base_dir, "test")
    
    # Create main directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subdirectories for each image type
    for dataset_dir in [train_dir, test_dir]:
        os.makedirs(os.path.join(dataset_dir, "faces"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "left_eyes"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "right_eyes"), exist_ok=True)
    
    return train_dir, test_dir

def split_dataset(input_dir, output_base_dir, test_size=0.3, random_state=42):
    """
    Split the dataset into training and test sets.
    
    Args:
        input_dir: Directory containing processed data
        output_base_dir: Directory to save train and test datasets
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
    """
    # Check for raw metadata first
    raw_metadata_path = os.path.join(input_dir, "metadata.json")
    processed_metadata_path = os.path.join(input_dir, "processed_metadata.json")
    
    if not os.path.exists(processed_metadata_path):
        if os.path.exists(raw_metadata_path):
            print(f"ERROR: Found raw metadata but not processed metadata at {input_dir}")
            print("Please run the metadata processing step first by using:")
            print("python src/preprocess_main.py --skip_extraction --processed_dir output")
            print("\nOr run the full pipeline with:")
            print("python src/preprocess_main.py --raw_data_dir dataset --processed_dir output --split_dir output/split_data")
        else:
            print(f"ERROR: No metadata found at {input_dir}")
            print("Please check your directory structure or run the extraction step first with:")
            print("python src/preprocess_main.py --raw_data_dir dataset --processed_dir output")
        return False
    
    # Load the processed metadata
    metadata = load_processed_metadata(processed_metadata_path)
    if not metadata:
        print(f"WARNING: Processed metadata at {processed_metadata_path} is empty")
        return False
    
    # Create output directories
    train_dir, test_dir = create_dataset_directories(output_base_dir)
    
    # Prepare for stratification if possible
    # Using dot positions as stratification targets
    try:
        # Extract dot numbers for stratification (assuming they exist in the metadata)
        dot_numbers = [entry.get("dot_data", {}).get("DotNum", 0) for entry in metadata]
        
        # If too many unique values or missing values, don't stratify
        if len(set(dot_numbers)) > len(metadata) // 10 or None in dot_numbers:
            stratify = None
        else:
            stratify = dot_numbers
    except (KeyError, TypeError):
        stratify = None
    
    # Split the metadata
    train_metadata, test_metadata = train_test_split(
        metadata, 
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    print(f"Split dataset: {len(train_metadata)} training samples, {len(test_metadata)} test samples")

    # Function to copy files based on metadata
    def copy_files_from_metadata(metadata_list, source_dir, dest_dir):
        for entry in metadata_list:
            # Copy face image
            if entry.get("face_image"):
                face_filename = os.path.basename(entry["face_image"])
                source = os.path.join(source_dir, "faces", face_filename)
                destination = os.path.join(dest_dir, "faces", face_filename)
                if os.path.exists(source):
                    shutil.copy2(source, destination)
            
            # Copy left eye image
            if entry.get("left_eye_image"):
                left_eye_filename = os.path.basename(entry["left_eye_image"])
                source = os.path.join(source_dir, "left_eyes", left_eye_filename)
                destination = os.path.join(dest_dir, "left_eyes", left_eye_filename)
                if os.path.exists(source):
                    shutil.copy2(source, destination)
            
            # Copy right eye image
            if entry.get("right_eye_image"):
                right_eye_filename = os.path.basename(entry["right_eye_image"])
                source = os.path.join(source_dir, "right_eyes", right_eye_filename)
                destination = os.path.join(dest_dir, "right_eyes", right_eye_filename)
                if os.path.exists(source):
                    shutil.copy2(source, destination)
    
    # Copy files to train and test directories
    copy_files_from_metadata(train_metadata, input_dir, train_dir)
    copy_files_from_metadata(test_metadata, input_dir, test_dir)
    
    # Save split metadata
    with open(os.path.join(train_dir, "metadata.json"), "w") as f:
        json.dump(train_metadata, f, indent=4)
    
    with open(os.path.join(test_dir, "metadata.json"), "w") as f:
        json.dump(test_metadata, f, indent=4)
    
    print(f"Dataset successfully split and saved to {output_base_dir}")
    return True


def process_all_subject_folders(input_base_dir, output_base_dir, test_size=0.3):
    """Process all subject folders in the input directory."""
    os.makedirs(output_base_dir, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    for folder in os.listdir(input_base_dir):
        folder_path = os.path.join(input_base_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing subject folder: {folder}")
            output_folder_path = os.path.join(output_base_dir, folder)
            success = split_dataset(folder_path, output_folder_path, test_size=test_size)
            if success:
                processed_count += 1
            else:
                skipped_count += 1
    
    print(f"\nSplitting complete: {processed_count} folders processed, {skipped_count} folders skipped")
    if skipped_count > 0:
        print("Some folders were skipped. Please ensure the metadata processing step was run successfully.")


if __name__ == "__main__":
    input_directory = os.path.join("output")  # Directory with processed data
    output_directory = os.path.join("split_data")  # Directory to save train/test data
    test_split = 0.3
    
    process_all_subject_folders(input_directory, output_directory, test_split)
