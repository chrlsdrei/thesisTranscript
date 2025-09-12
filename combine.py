#!/usr/bin/env python3
"""
Script to combine all text files from final_text_output folder into a single corpus.txt file.
"""

import os
import glob

def combine_text_files():
    """
    Combine all text files from final_text_output folder into corpus.txt
    """
    # Define paths
    input_folder = "final_text_output"
    output_file = "corpus.txt"
    
    # Get all text files from the input folder
    text_files = glob.glob(os.path.join(input_folder, "*_text"))
    
    # Sort files to ensure consistent ordering
    text_files.sort()
    
    print(f"Found {len(text_files)} text files to combine:")
    for file in text_files:
        print(f"  - {file}")
    
    # Combine all files into corpus.txt
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in text_files:
            print(f"Processing: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    # Write content without adding extra newlines
                    outfile.write(content)
                    # Only add newline if the content doesn't end with one
                    if content and not content.endswith('\n'):
                        outfile.write('\n')
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"\nSuccessfully combined all files into {output_file}")

if __name__ == "__main__":
    combine_text_files()