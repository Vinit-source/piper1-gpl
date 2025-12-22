#!/usr/bin/env python3
"""
Resample audio files to 22050 Hz using librosa.
Reads from content/dataset_24hr/wavs
Writes to content/dataset_24hr/wavs_22050
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import librosa
import soundfile as sf
import argparse
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path("/Users/vinitgore/Documents/NavGurukul/piper1-gpl/content/dataset_24hr")
INPUT_DIR = BASE_DIR / "wavs"
OUTPUT_DIR = BASE_DIR / "wavs_22050"
TARGET_SR = 22050

def resample_file(args):
    """
    Resample a single file.
    Args:
        args: tuple (input_path, output_path, target_sr)
    Returns:
        tuple (success: bool, message: str)
    """
    input_path, output_path, target_sr = args
    
    try:
        # Load audio with target sample rate
        # librosa.load resamples automatically if sr is provided
        y, s = librosa.load(input_path, sr=target_sr)
        
        # Write to output
        sf.write(output_path, y, target_sr)
        return True, None
    except Exception as e:
        return False, f"Error processing {input_path.name}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Resample wav files to 22050 Hz")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return

    logger.info(f"Input Directory: {INPUT_DIR}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Target Sample Rate: {TARGET_SR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # List all wav files
    wav_files = list(INPUT_DIR.glob("*.wav"))
    logger.info(f"Found {len(wav_files)} wav files.")

    if not wav_files:
        logger.warning("No wav files found.")
        return

    # Prepare tasks
    tasks = []
    for wav_file in wav_files:
        output_file = OUTPUT_DIR / wav_file.name
        tasks.append((wav_file, output_file, TARGET_SR))

    # Process
    logger.info(f"Starting resampling with {args.workers} workers...")
    
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Map tasks
        results = list(tqdm(executor.map(resample_file, tasks), total=len(tasks), unit="file"))
        
        for success, message in results:
            if success:
                success_count += 1
            else:
                error_count += 1
                logger.error(message)

    logger.info("="*40)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Successfully resampled: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output saved to: {OUTPUT_DIR}")
    logger.info("="*40)

if __name__ == "__main__":
    main()
