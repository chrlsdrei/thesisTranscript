#!/usr/bin/env python3
"""
process_kaldi.py - Convert transcription files to Kaldi-readable format

This script converts transcription files from 'new_full_transcript' folder
into two output formats: text files and segment files for Kaldi ASR system.
"""

import os
import re
from pathlib import Path


def clean_text(text):
    """
    Clean text by converting to lowercase and removing all punctuation.
    Keep only words and spaces.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text with only lowercase words and spaces
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove all punctuation and special characters, keep only letters, numbers, and spaces
    # This will remove . , ? ! : ; " ' - and other symbols
    text = re.sub(r'[^\w\s]', '', text)
    
    # Replace multiple spaces with single space and strip
    text = ' '.join(text.split())
    
    return text


def parse_transcript_file(filepath):
    """
    Parse a transcript file and extract utterance information.
    
    Args:
        filepath (str): Path to the transcript file
        
    Returns:
        list: List of dictionaries containing utterance data
    """
    utterances = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for "Utterance" line
        if line.startswith('Utterance'):
            utterance_data = {}
            
            # Get Start time
            i += 1
            if i < len(lines) and lines[i].strip().startswith('Start:'):
                start_time = lines[i].strip().split(':', 1)[1].strip()
                utterance_data['start'] = start_time
            
            # Get End time
            i += 1
            if i < len(lines) and lines[i].strip().startswith('End:'):
                end_time = lines[i].strip().split(':', 1)[1].strip()
                utterance_data['end'] = end_time
            
            # Skip Confidence line
            i += 1
            if i < len(lines) and lines[i].strip().startswith('Confidence:'):
                i += 1
            
            # Get Text
            if i < len(lines) and lines[i].strip().startswith('Text:'):
                text = lines[i].strip().split(':', 1)[1].strip()
                utterance_data['text'] = clean_text(text)
                utterances.append(utterance_data)
        
        i += 1
    
    return utterances


def process_file(input_filepath, text_output_dir, segments_output_dir):
    """
    Process a single transcript file and create corresponding text and segments files.
    
    Args:
        input_filepath (Path): Path to the input transcript file
        text_output_dir (Path): Directory for text output files
        segments_output_dir (Path): Directory for segments output files
    """
    # Extract base filename (remove _full_transcript suffix)
    filename = input_filepath.stem  # Get filename without extension
    if filename.endswith('_full_transcript'):
        base_name = filename[:-len('_full_transcript')]
    else:
        base_name = filename
    
    # Parse the transcript file
    utterances = parse_transcript_file(input_filepath)
    
    if not utterances:
        print(f"Warning: No utterances found in {input_filepath}")
        return
    
    # Create output filenames
    text_output_file = text_output_dir / f"{base_name}_text"
    segments_output_file = segments_output_dir / f"{base_name}_segments"
    
    # Write text file
    with open(text_output_file, 'w', encoding='utf-8') as f:
        for idx, utt in enumerate(utterances, 1):
            utterance_id = f"{base_name}_{idx:04d}"
            f.write(f"{utterance_id} {utt['text']}\n")
    
    # Write segments file
    with open(segments_output_file, 'w', encoding='utf-8') as f:
        for idx, utt in enumerate(utterances, 1):
            utterance_id = f"{base_name}_{idx:04d}"
            f.write(f"{utterance_id} {base_name} {utt['start']} {utt['end']}\n")
    
    print(f"Processed: {input_filepath.name}")
    print(f"  -> {text_output_file.name}")
    print(f"  -> {segments_output_file.name}")


def main():
    """
    Main function to process all transcript files.
    """
    # Define directories
    input_dir = Path('new_full_transcript')
    text_output_dir = Path('final_text_output')
    segments_output_dir = Path('final_segments_output')
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        print("Please make sure the 'new_full_transcript' folder exists in the current directory.")
        return
    
    # Create output directories if they don't exist
    text_output_dir.mkdir(exist_ok=True)
    segments_output_dir.mkdir(exist_ok=True)
    
    # Get all transcript files
    transcript_files = list(input_dir.glob('*'))
    
    if not transcript_files:
        print(f"No files found in '{input_dir}' directory!")
        return
    
    print(f"Found {len(transcript_files)} files to process")
    print("-" * 50)
    
    # Process each file
    successful = 0
    failed = 0
    
    for filepath in transcript_files:
        if filepath.is_file():
            try:
                process_file(filepath, text_output_dir, segments_output_dir)
                successful += 1
            except Exception as e:
                print(f"Error processing {filepath.name}: {str(e)}")
                failed += 1
        else:
            print(f"Skipping non-file: {filepath.name}")
    
    # Print summary
    print("-" * 50)
    print(f"Processing complete!")
    print(f"  Successfully processed: {successful} files")
    if failed > 0:
        print(f"  Failed: {failed} files")
    print(f"\nOutput directories:")
    print(f"  Text files: {text_output_dir}/")
    print(f"  Segment files: {segments_output_dir}/")


if __name__ == "__main__":
    main()