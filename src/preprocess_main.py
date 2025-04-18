import os
import sys
import argparse
from pre_processing.extraction import process_all_folders
from pre_processing.process_metadata import process_all_folders as process_all_metadata

def main():
    parser = argparse.ArgumentParser(description="Eye Gaze Dataset Processing Pipeline")
    parser.add_argument("--raw_data_dir", type=str, default="dataset", help="Directory containing raw data")
    parser.add_argument("--processed_dir", type=str, default="output", help="Directory to save processed data")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip extraction step")
    parser.add_argument("--skip_metadata_processing", action="store_true", help="Skip metadata processing step")
    
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
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()