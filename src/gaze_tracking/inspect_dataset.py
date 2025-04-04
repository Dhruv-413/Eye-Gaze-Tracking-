#!/usr/bin/env python3
"""
Utility script to inspect and validate a gaze tracking dataset.
"""

import os
import json
import argparse
import sys
from typing import Dict, List, Any

def inspect_metadata_file(metadata_path: str) -> None:
    """
    Inspect a metadata JSON file and report its structure.
    
    Args:
        metadata_path: Path to the metadata file
    """
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found: {metadata_path}")
        return
        
    try:
        with open(metadata_path, 'r') as file:
            data = json.load(file)
            
        print(f"\nMetadata file: {metadata_path}")
        print(f"Number of frames: {len(data)}")
        
        # Check first frame to understand structure
        if data:
            print("\nFirst frame keys:")
            first_frame = data[0]
            for key in first_frame.keys():
                print(f"  - {key}")
                
            # Sample one frame to check structure in detail
            print("\nSample frame structure:")
            for key, value in first_frame.items():
                if isinstance(value, dict):
                    print(f"  {key}: {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"  {key}: List with {len(value)} items")
                    if value and len(value) < 5:
                        print(f"    Items: {value}")
                else:
                    print(f"  {key}: {value}")
                    
            # Check if important fields exist
            has_face_images = 'face_images' in first_frame
            has_face_image = 'face_image' in first_frame
            has_dot_data = 'dot_data' in first_frame
            
            if has_dot_data:
                dot_data = first_frame['dot_data']
                has_normalized_coords = 'normalized_X' in dot_data and 'normalized_Y' in dot_data
                
                if has_normalized_coords:
                    print("\nGaze coordinates found:")
                    print(f"  X: {dot_data['normalized_X']}")
                    print(f"  Y: {dot_data['normalized_Y']}")
                else:
                    print("\nWARNING: Missing normalized coordinates in dot_data")
            else:
                print("\nWARNING: Missing dot_data field")
                
            # Check face image paths
            face_path = None
            if has_face_images and first_frame['face_images']:
                face_path = first_frame['face_images'][0]
                print(f"\nFace image path: {face_path}")
            elif has_face_image:
                face_path = first_frame['face_image']
                print(f"\nFace image path: {face_path}")
            else:
                print("\nWARNING: No face image path found")
                
            # Check if the face image exists
            if face_path:
                # Try both as absolute path and relative to metadata directory
                if os.path.exists(face_path):
                    print(f"Face image exists: Yes")
                else:
                    base_dir = os.path.dirname(metadata_path)
                    rel_path = os.path.join(base_dir, face_path)
                    if os.path.exists(rel_path):
                        print(f"Face image exists: Yes (as relative path)")
                    else:
                        print(f"Face image exists: No")
                        print(f"Checked absolute path: {face_path}")
                        print(f"Checked relative path: {rel_path}")
    except Exception as e:
        print(f"ERROR: Failed to inspect metadata file: {e}")

def inspect_directory_structure(dataset_dir: str) -> None:
    """
    Inspect the directory structure of a dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
    """
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return
        
    print(f"\nInspecting directory: {dataset_dir}")
    
    # Count files in each subdirectory
    for root, dirs, files in os.walk(dataset_dir):
        rel_path = os.path.relpath(root, dataset_dir)
        if rel_path == '.':
            print(f"Root directory: {len(files)} files, {len(dirs)} subdirectories")
        else:
            print(f"Subdirectory: {rel_path}")
            print(f"  Files: {len(files)}")
            
        # Show sample files
        if files:
            sample_files = files[:5] if len(files) > 5 else files
            print(f"  Sample files: {', '.join(sample_files)}")

def extract_frame_id(path: str) -> str:
    """
    Extract the frame ID from a file path.
    
    Args:
        path: Path to extract ID from
        
    Returns:
        Frame ID or None if not found
    """
    # Get filename without extension
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    
    # Split by underscore and look for numbers
    parts = name.split('_')
    for part in parts:
        if part.isdigit():
            return part
        elif part.isalnum() and any(c.isdigit() for c in part):
            # Handle alphanumeric IDs
            return part
            
    # If no clear ID found, try to extract digits
    digits = ''.join(c for c in name if c.isdigit())
    if digits:
        return digits
        
    return None

def fix_metadata_paths(metadata_path: str, data: List[Dict[str, Any]], dataset_dir: str) -> None:
    """
    Attempt to fix face image paths in the metadata file.
    
    Args:
        metadata_path: Path to the metadata file
        data: Loaded metadata data
        dataset_dir: Path to the dataset directory
    """
    print("\nAttempting to fix metadata paths...")
    
    # Find face image directories
    face_dirs = []
    for root, dirs, files in os.walk(dataset_dir):
        for directory in dirs:
            if "face" in directory.lower():
                face_dirs.append(os.path.join(root, directory))
    
    if not face_dirs:
        print("No face image directories found.")
        return
        
    print(f"Found {len(face_dirs)} potential face image directories:")
    for i, directory in enumerate(face_dirs, 1):
        print(f"{i}. {directory}")
        
    # Ask user to select a directory
    try:
        choice = int(input("\nEnter the number of the face image directory to use (or 0 to exit): "))
        if not (1 <= choice <= len(face_dirs)):
            print("Exiting...")
            return
            
        face_dir = face_dirs[choice - 1]
    except ValueError:
        print("Invalid input. Exiting...")
        return
    
    # Get all face images in the selected directory
    face_images = {}
    for file in os.listdir(face_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Extract frame number or identifier from filename
            # Example: assuming filenames like "frame_00001_face.jpg" or "00001_face.jpg"
            parts = file.split('_')
            for part in parts:
                # Try to find a number part
                if part.isdigit():
                    face_images[part] = os.path.join(face_dir, file)
                    break
                elif part.isalnum() and any(c.isdigit() for c in part):
                    # Handle alphanumeric IDs
                    face_images[part] = os.path.join(face_dir, file)
                    break
    
    if not face_images:
        print("No face images found with identifiable frame numbers.")
        return
        
    print(f"Found {len(face_images)} face images with identifiable frame numbers.")
    
    # Now try to match and fix paths in the metadata
    fixed_count = 0
    for i, frame in enumerate(data):
        # Try to find a suitable face image for this frame
        frame_id = None
        
        # Extract frame ID from existing path if possible
        if 'face_images' in frame and frame['face_images']:
            old_path = frame['face_images'][0]
            frame_id = extract_frame_id(old_path)
        elif 'face_image' in frame:
            old_path = frame['face_image']
            frame_id = extract_frame_id(old_path)
        elif 'frame_path' in frame:
            old_path = frame['frame_path']
            frame_id = extract_frame_id(old_path)
            
        # If we found a frame ID, try to match it
        if frame_id and frame_id in face_images:
            # Update the path
            new_path = face_images[frame_id]
            rel_path = os.path.relpath(new_path, os.path.dirname(metadata_path))
            
            if 'face_images' in frame:
                frame['face_images'] = [rel_path]
            else:
                frame['face_image'] = rel_path
                
            fixed_count += 1
            
        # Show progress
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(data)} frames...")
    
    print(f"Fixed {fixed_count} paths out of {len(data)} frames.")
    
    if fixed_count > 0:
        # Ask user if they want to save the fixed metadata
        save = input("\nDo you want to save the fixed metadata? (y/n): ").strip().lower()
        if save == 'y':
            backup_path = metadata_path + '.backup'
            # Create a backup first
            try:
                with open(backup_path, 'w') as backup_file:
                    json.dump(data, backup_file, indent=2)
                print(f"Created backup at {backup_path}")
                
                # Save the fixed metadata
                with open(metadata_path, 'w') as fixed_file:
                    json.dump(data, fixed_file, indent=2)
                print(f"Saved fixed metadata to {metadata_path}")
            except Exception as e:
                print(f"ERROR: Failed to save fixed metadata: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect a gaze tracking dataset")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--check_paths', action='store_true',
                        help='Check if all face image paths exist')
    parser.add_argument('--fix_metadata', action='store_true',
                        help='Attempt to fix metadata file by updating paths')
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return
        
    # Find metadata file
    metadata_path = os.path.join(dataset_dir, 'processed_metadata.json')
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found: {metadata_path}")
        print("Searching for other JSON files...")
        
        json_files = []
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
                    
        if json_files:
            print(f"Found {len(json_files)} JSON files:")
            for i, file_path in enumerate(json_files, 1):
                print(f"{i}. {file_path}")
                
            # Ask user to select a file
            try:
                choice = int(input("\nEnter the number of the metadata file to inspect (or 0 to exit): "))
                if 1 <= choice <= len(json_files):
                    metadata_path = json_files[choice - 1]
                else:
                    print("Exiting...")
                    return
            except ValueError:
                print("Invalid input. Exiting...")
                return
        else:
            print("No JSON files found in the dataset directory.")
            
            # Inspect directory structure anyway
            inspect_directory_structure(dataset_dir)
            return
    
    # Inspect metadata file
    inspect_metadata_file(metadata_path)
    
    # Inspect directory structure
    inspect_directory_structure(dataset_dir)
    
    # Check all paths if requested
    if args.check_paths:
        try:
            with open(metadata_path, 'r') as file:
                data = json.load(file)
                
            total_frames = len(data)
            missing_paths = 0
            
            print(f"\nChecking {total_frames} face image paths...")
            
            for i, frame in enumerate(data):
                face_path = None
                if 'face_images' in frame and frame['face_images']:
                    face_path = frame['face_images'][0]
                elif 'face_image' in frame:
                    face_path = frame['face_image']
                    
                if face_path:
                    # Check as absolute path
                    exists = os.path.exists(face_path)
                    
                    # If not, check as relative path
                    if not exists:
                        rel_path = os.path.join(os.path.dirname(metadata_path), face_path)
                        exists = os.path.exists(rel_path)
                        
                    if not exists:
                        missing_paths += 1
                        if missing_paths <= 5:  # Show only the first few missing paths
                            print(f"Missing path: {face_path}")
                else:
                    missing_paths += 1
                    
                # Show progress
                if (i+1) % 100 == 0:
                    print(f"Checked {i+1}/{total_frames} frames...")
                    
            print(f"\nFound {missing_paths} missing paths out of {total_frames} frames.")
            
            # Try to fix paths if requested
            if args.fix_metadata and missing_paths > 0:
                fix_metadata_paths(metadata_path, data, dataset_dir)
                
        except Exception as e:
            print(f"ERROR: Failed to check paths: {e}")

if __name__ == "__main__":
    main()
