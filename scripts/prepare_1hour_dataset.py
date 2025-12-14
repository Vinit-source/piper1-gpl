#!/usr/bin/env python3
"""
Prepare an optimal 8-hour subset from the SPICOR dataset.
This script selects samples that maximize phoneme coverage (specifically Indian English)
and prioritizes transcripts containing non-standard English words (Indian proper nouns, etc.)
using NLTK to filter out standard US English words.
"""

import json
import os
import subprocess
import wave
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import random

# Try to import nltk, download resources if needed
try:
    import nltk
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download('words')
    from nltk.corpus import words
    US_ENGLISH_VOCAB = set(w.lower() for w in words.words())
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not found. Please install it with 'pip install nltk'.")
    print("Falling back to basic filtering.")
    US_ENGLISH_VOCAB = set()
    NLTK_AVAILABLE = False

# Configuration
TARGET_DURATION_HOURS = 1
TARGET_DURATION_SECONDS = TARGET_DURATION_HOURS * 3600
TARGET_WORD_COUNT = 8520  # Approx words for 1 hour
SAMPLE_RATE = 22050

# Base paths
BASE_DIR = Path("/Users/vinitgore/Documents/NavGurukul/piper1-gpl")
SPICOR_DIR = BASE_DIR / "content/dataset_8hr"
METADATA_FILE = SPICOR_DIR / "metadata.csv"
WAV_DIR = SPICOR_DIR / "wavs"
OUTPUT_DIR = BASE_DIR / "content/dataset_1hr"
CONFIG_JSON = BASE_DIR / "configs/en_US_HFC_Female_from_hi.json"

@dataclass
class Sample:
    """Represents a single audio sample."""
    id: str
    transcript: str
    domain: str
    wav_path: Path
    duration: float = 0.0
    phonemes: str = ""
    phoneme_set: Set[str] = None
    pronoun_count: int = 0
    indian_term_count: int = 0
    word_count: int = 0
    non_english_words: List[str] = None
    
    def __post_init__(self):
        if self.phoneme_set is None:
            self.phoneme_set = set()
        if self.non_english_words is None:
            self.non_english_words = []

def load_target_phonemes() -> Set[str]:
    """Load target phonemes from the config file."""
    print(f"Loading target phonemes from {CONFIG_JSON}...")
    try:
        with open(CONFIG_JSON, 'r') as f:
            config = json.load(f)
        phoneme_map = config.get('phoneme_id_map', {})
        target_phonemes = set(phoneme_map.keys())
        print(f"Loaded {len(target_phonemes)} target phonemes.")
        return target_phonemes
    except Exception as e:
        print(f"Error loading config: {e}")
        return set()

def get_wav_duration(wav_path: Path) -> float:
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(str(wav_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        # print(f"Error reading {wav_path}: {e}")
        return 0.0

def phonemize_text(text: str) -> str:
    """Convert text to phonemes using espeak-ng."""
    try:
        # Use en-us as base, but we want to capture phonemes that map to our target set
        result = subprocess.run(
            ["espeak-ng", "--ipa", "-q", "-v", "en-us", text],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error phonemizing: {e}")
        return ""

def extract_phoneme_set(phonemes: str) -> Set[str]:
    """Extract individual phonemes from IPA string."""
    phoneme_set = set()
    clean_phonemes = phonemes.replace('ˈ', '').replace('ˌ', '').replace('ː', '').replace('͡', '')
    for char in clean_phonemes:
        if not char.isspace():
            phoneme_set.add(char)
    return phoneme_set

def clean_and_tokenize(text: str) -> List[str]:
    """Converts text to 'no punctuation no case' and splits into words."""
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()
    return clean_text.split()

def get_non_english_words(text: str) -> List[str]:
    """Returns a list of words from the text that are NOT in the English dictionary."""
    if not NLTK_AVAILABLE:
        return []
        
    tokens = clean_and_tokenize(text)
    non_english = []
    for word in tokens:
        # Check if word is in dictionary
        # Also ignore pure numbers
        if word not in US_ENGLISH_VOCAB and not word.isdigit():
            non_english.append(word)
    return non_english

def load_dataset() -> Dict[str, Sample]:
    """Load the transcript from metadata.csv and create Sample objects."""
    print(f"Loading transcripts from {METADATA_FILE}...")
    
    samples = {}
    missing_wavs = 0
    
    with open(METADATA_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
            
            sample_id = parts[0]
            transcript_text = parts[1]
            
            # Infer domain from filename (e.g., IISc_SPICORProject_EN_F_AGRI_1821 -> AGRI)
            # Format seems to be PREFIX_DOMAIN_ID
            try:
                # Split by underscore
                id_parts = sample_id.split('_')
                # Assuming format IISc_SPICORProject_EN_F_{DOMAIN}_{ID}
                # Index 4 should be domain if format holds
                if len(id_parts) >= 5:
                    domain = id_parts[4]
                else:
                    domain = "UNKNOWN"
            except:
                domain = "UNKNOWN"

            wav_path = WAV_DIR / f"{sample_id}.wav"
            
            if not wav_path.exists():
                missing_wavs += 1
                continue
            
            samples[sample_id] = Sample(
                id=sample_id,
                transcript=transcript_text,
                domain=domain,
                wav_path=wav_path,
                word_count=len(transcript_text.split())
            )
    
    print(f"Loaded {len(samples)} samples ({missing_wavs} missing wav files)")
    return samples

def analyze_samples(samples: Dict[str, Sample], target_phonemes: Set[str]) -> None:
    """Analyze samples for duration, phonemes, and features."""
    print("\nAnalyzing samples...")
    total = len(samples)
    
    for i, (sample_id, sample) in enumerate(samples.items()):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{total} samples...")
        
        # Get duration
        sample.duration = get_wav_duration(sample.wav_path)
        
        # Get phonemes
        sample.phonemes = phonemize_text(sample.transcript)
        sample.phoneme_set = extract_phoneme_set(sample.phonemes)
        
        if target_phonemes:
            sample.phoneme_set = sample.phoneme_set.intersection(target_phonemes)
        
        # Identify non-English words
        sample.non_english_words = get_non_english_words(sample.transcript)
    
    # Print statistics
    total_duration = sum(s.duration for s in samples.values())
    print(f"\nTotal dataset duration: {total_duration/3600:.2f} hours")
    if len(samples) > 0:
        print(f"Average sample duration: {total_duration/len(samples):.2f} seconds")
    
    # Phoneme statistics
    all_phonemes = set()
    for sample in samples.values():
        all_phonemes.update(sample.phoneme_set)
    print(f"Total unique target phonemes found in dataset: {len(all_phonemes)}")

def greedy_selection(samples: Dict[str, Sample], target_duration: float, target_phonemes: Set[str]) -> List[Sample]:
    """
    Select samples using a greedy algorithm.
    Prioritizes samples with non-English words first, then fills with other high-value samples.
    Ensures equal representation of domains where possible.
    """
    print("\nSelecting samples...")
    
    # Calculate phoneme frequencies across all samples
    phoneme_freq = Counter()
    for sample in samples.values():
        phoneme_freq.update(sample.phoneme_set)
    
    # Group samples by domain
    samples_by_domain = defaultdict(list)
    for sample in samples.values():
        samples_by_domain[sample.domain].append(sample)
    
    domains = sorted(samples_by_domain.keys())
    print(f"Found {len(domains)} domains: {domains}")
    
    # Calculate target words per domain
    target_per_domain = TARGET_WORD_COUNT / len(domains)
    print(f"Target words per domain: {target_per_domain:.0f}")
    
    # Calculate score for each sample
    def sample_score(sample: Sample, covered_phonemes: Set[str]) -> float:
        # New phonemes this sample would add
        new_phonemes = sample.phoneme_set - covered_phonemes
        
        # Weight rare phonemes more heavily
        phoneme_score = 0
        for p in new_phonemes:
            phoneme_score += 10.0 / (phoneme_freq[p] + 1)
        
        repetition_score = 0
        for p in sample.phoneme_set:
             repetition_score += 0.1
        
        pronoun_bonus = sample.pronoun_count * 1.0
        indian_bonus = sample.indian_term_count * 5.0
        
        # Bonus for non-English words count
        non_english_bonus = len(sample.non_english_words) * 10.0
        
        duration_penalty = 0
        if sample.duration < 2.0:
            duration_penalty = 2.0
        elif sample.duration > 15.0:
            duration_penalty = 1.0
            
        return phoneme_score + repetition_score + pronoun_bonus + indian_bonus + non_english_bonus - duration_penalty
    
    selected = []
    covered_phonemes = set()
    total_selected_duration = 0.0
    total_words = 0
    selected_ids = set()
    
    # --- Phase 1: Select samples per domain to meet quota ---
    print("Phase 1: Selecting samples per domain to meet quota...")
    
    for domain in domains:
        domain_samples = samples_by_domain[domain]
        
        # Sort samples by score (prioritizing non-English words via score)
        # We use a static score for sorting to avoid re-calculating every time
        # For the static sort, we assume covered_phonemes is empty or just use the static parts
        # But let's just use the full score with current covered_phonemes? 
        # No, that changes as we select.
        # Let's sort by (Non-English Count, Indian Terms, Duration) as a proxy for "Best"
        domain_samples.sort(key=lambda s: (len(s.non_english_words), s.indian_term_count, -s.duration), reverse=True)
        
        domain_words = 0
        domain_selected_count = 0
        
        for sample in domain_samples:
            if sample.duration < 0.5: continue
            
            # Check if we reached quota for this domain
            if domain_words >= target_per_domain:
                break
            
            selected.append(sample)
            selected_ids.add(sample.id)
            covered_phonemes.update(sample.phoneme_set)
            total_selected_duration += sample.duration
            total_words += sample.word_count
            domain_words += sample.word_count
            domain_selected_count += 1
            
        print(f"  Domain {domain:<10}: Selected {domain_selected_count} samples, {domain_words} words (Target: {target_per_domain:.0f})")

    print(f"After Phase 1: {len(selected)} samples, {total_selected_duration/3600:.2f}h, {total_words} words")
    
    # --- Phase 2: Fill remaining global capacity with best remaining samples ---
    if total_words < TARGET_WORD_COUNT:
        print("Phase 2: Filling remaining global capacity with best remaining samples...")
        
        # Collect all remaining samples
        remaining_samples = []
        for sample in samples.values():
            if sample.id not in selected_ids:
                remaining_samples.append(sample)
        
        # Sort remaining by score (Non-English first, then others)
        remaining_samples.sort(key=lambda s: (len(s.non_english_words), s.indian_term_count, -s.duration), reverse=True)
        
        for sample in remaining_samples:
            if total_words >= TARGET_WORD_COUNT:
                break
                
            if sample.duration < 0.5: continue
            
            selected.append(sample)
            selected_ids.add(sample.id)
            covered_phonemes.update(sample.phoneme_set)
            total_selected_duration += sample.duration
            total_words += sample.word_count
            
            if len(selected) % 500 == 0:
                print(f"  Selected {len(selected)} samples, {total_selected_duration/3600:.2f}h, {total_words} words")

    return selected

def save_selected_dataset(selected: List[Sample]) -> None:
    """Save the selected dataset as metadata.csv and copy wav files."""
    import shutil
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_output_dir = OUTPUT_DIR / "wavs"
    wav_output_dir.mkdir(exist_ok=True)
    
    # Write metadata.csv
    metadata_path = OUTPUT_DIR / "metadata.csv"
    with open(metadata_path, 'w') as f:
        for sample in selected:
            # Format: filename|text (without .wav extension)
            filename = sample.wav_path.stem
            f.write(f"{filename}|{sample.transcript}\n")
    
    print(f"\nWritten metadata to {metadata_path}")
    
    # Copy wav files
    print(f"Copying {len(selected)} wav files to {wav_output_dir}...")
    
    count = 0
    for i, sample in enumerate(selected):
        if (i + 1) % 500 == 0:
            print(f"  Copied {i + 1}/{len(selected)} files...")
        
        dest_path = wav_output_dir / sample.wav_path.name
        if not dest_path.exists():
            try:
                shutil.copy2(sample.wav_path, dest_path)
                count += 1
            except Exception as e:
                print(f"Error copying {sample.wav_path}: {e}")
    
    print(f"Done! Copied {count} files. Dataset saved to {OUTPUT_DIR}")

def print_stats(selected: List[Sample], target_phonemes: Set[str]):
    total_duration = sum(s.duration for s in selected)
    total_words = sum(s.word_count for s in selected)
    
    print("\n" + "="*60)
    print("FINAL DATASET STATISTICS")
    print("="*60)
    print(f"Total Samples: {len(selected)}")
    print(f"Total Duration: {total_duration/3600:.2f} hours")
    print(f"Total Words: {total_words}")
    
    covered = set()
    non_english_sample_count = 0
    for s in selected:
        covered.update(s.phoneme_set)
        if len(s.non_english_words) > 0:
            non_english_sample_count += 1
            
    print(f"Samples with non-English words: {non_english_sample_count} ({non_english_sample_count/len(selected)*100:.1f}%)")
    
    if target_phonemes:
        print(f"Phoneme Coverage: {len(covered)}/{len(target_phonemes)} ({len(covered)/len(target_phonemes)*100:.1f}%)")
        missing = target_phonemes - covered
        if missing:
            print(f"Missing Phonemes: {missing}")

def main():
    print("="*60)
    print("PIPER TTS - 1-Hour Dataset Preparation (Subset of 8-Hour)")
    print("="*60)
    
    target_phonemes = load_target_phonemes()
    samples = load_dataset()
    
    # Filter out samples with missing wavs
    samples = {k: v for k, v in samples.items() if v.wav_path.exists()}
    
    analyze_samples(samples, target_phonemes)
    
    selected = greedy_selection(samples, TARGET_DURATION_SECONDS, target_phonemes)
    
    print_stats(selected, target_phonemes)
    
    save_selected_dataset(selected)

if __name__ == "__main__":
    main()
