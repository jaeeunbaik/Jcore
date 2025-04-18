#!/usr/bin/env python3
import os
import glob
import re

def extract_transcription(line):
    """Extract transcription from various SCP file formats"""
    line = line.strip()
    
    # Format: utterance_id /path/to/audio.wav text transcription
    # Check for 3+ space-separated parts where the middle part has .wav, .flac, etc.
    parts = line.split()
    if len(parts) >= 3:
        for i, part in enumerate(parts[:-1]):
            if re.search(r'\.(wav|flac|mp3|sph)$', part):
                return ' '.join(parts[i+1:])
    
    # Format: /path/to/audio.wav|text transcription
    if '|' in line:
        return line.split('|', 1)[1]
    
    # Format: utterance_id text transcription (no audio path)
    parts = line.split(maxsplit=1)
    if len(parts) == 2 and not re.search(r'\.(wav|flac|mp3|sph)$', parts[0]):
        return parts[1]
    
    # If we can't determine format, return everything after first path-like part
    for i, part in enumerate(parts):
        if '/' in part:
            return ' '.join(parts[i+1:]) if i+1 < len(parts) else ""
    
    return ""  # Return empty if we can't extract transcription

def find_scp_files(directory="."):
    """
    Find all SCP files in the specified directory
    
    Args:
        directory (str): Path to directory to search for SCP files. Defaults to current directory.
    
    Returns:
        list: List of found SCP files with their paths
    """
    # Handle both relative and absolute paths
    search_path = os.path.join(directory, "*.scp")
    scp_files = glob.glob(search_path)
    
    if not scp_files:
        print(f"No SCP files found in directory: {directory}")
        return []
    
    print(f"Found {len(scp_files)} SCP files in {directory}")
    return scp_files


def process_scp_file(scp_file):
    """Process a single SCP file and extract transcriptions"""
    print(f"Processing {scp_file}...")
    file_transcriptions = []
    
    with open(scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            transcription = extract_transcription(line)
            if transcription:
                file_transcriptions.append(transcription)
    
    print(f"  - Extracted {len(file_transcriptions)} transcriptions.")
    return file_transcriptions

def write_transcriptions(transcriptions, output_file="all_transcriptions.txt"):
    """Write all transcriptions to output file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for transcription in transcriptions:
            f.write(f"{transcription}\n")
    
    print(f"Successfully wrote {len(transcriptions)} transcriptions to {output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract transcriptions from SCP files")
    parser.add_argument("--dir", default="/home/hdd2/jenny/Self-Distillation-ASR/scp", help="Directory containing SCP files")
    parser.add_argument("--output", default="/home/hdd2/jenny/Self-Distillation-ASR/utils/spm/input.txt", help="Output file name")
    args = parser.parse_args()
    
    scp_files = find_scp_files(args.dir)
    if not scp_files:
        return
    
    transcriptions = []
    for scp_file in scp_files:
        file_transcriptions = process_scp_file(scp_file)
        transcriptions.extend(file_transcriptions)
    
    write_transcriptions(transcriptions, args.output)

if __name__ == "__main__":
    main()