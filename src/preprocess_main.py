import os
import sys
import argparse
from pre_processing.extraction import process_all_folders
from pre_processing.process_metadata import process_all_folders as process_all_metadata
from pre_processing.data_splitter import process_all_subject_folders

def main():
    parser = argparse.ArgumentParser(description="Eye Gaze Dataset Processing Pipeline")
    parser.add_argument("--raw_data_dir", type=str, default="dataset", help="Directory containing raw data")
    parser.add_argument("--processed_dir", type=str, default="output", help="Directory to save processed data")
    parser.add_argument("--split_dir", type=str, default="output/split_data", help="Directory to save train/test split data")
    parser.add_argument("--test_size", type=float, default=0.3, help="Proportion of data to use for testing")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip extraction step")
    parser.add_argument("--skip_metadata_processing", action="store_true", help="Skip metadata processing step")
    parser.add_argument("--skip_splitting", action="store_true", help="Skip train/test splitting step")
    
    args = parser.parse_args()
    
    # Step 1: Extract images and annotations from raw data
    if not args.skip_extraction:
        print("Step 1: Extracting images and annotations...")
        if not os.path.exists(args.raw_data_dir):
            print(f"Error: Raw data directory '{args.raw_data_dir}' not found.")
            sys.exit(1)
        process_all_folders(args.raw_data_dir, args.processed_dir)
        print("Extraction complete!\n")
    else:
        print("Skipping extraction step.\n")
    
    # Step 2: Process metadata (filter invalid frames, normalize coordinates, etc.)
    if not args.skip_metadata_processing:
        print("Step 2: Processing metadata...")
        if not os.path.exists(args.processed_dir):
            print(f"Error: Processed data directory '{args.processed_dir}' not found.")
            sys.exit(1)
        process_all_metadata(args.processed_dir)
        print("Metadata processing complete!\n")
    else:
        print("Skipping metadata processing step.\n")
    
    # Step 3: Split data into train and test sets
    if not args.skip_splitting:
        print("Step 3: Splitting data into train and test sets...")
        if not os.path.exists(args.processed_dir):
            print(f"Error: Processed data directory '{args.processed_dir}' not found.")
            sys.exit(1)
        process_all_subject_folders(args.processed_dir, args.split_dir, args.test_size)
        print("Data splitting complete!\n")
    else:
        print("Skipping data splitting step.\n")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
