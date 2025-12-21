#!/usr/bin/env python3
"""
Prepare a 24-hour subset from the SPICOR dataset.
Requirements:
1. 3/5 representation for non-English words.
2. Equal representation for all categories (domains).
3. Target destination: content/dataset_24hr.
4. Approximate target word count: 204480.
"""

import json
import os
import subprocess
import wave
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
TARGET_DURATION_HOURS = 24
TARGET_DURATION_SECONDS = TARGET_DURATION_HOURS * 3600
TARGET_WORD_COUNT = 204480  # Approx words for 24 hours
SAMPLE_RATE = 22050

# Base paths
BASE_DIR = Path("/Users/vinitgore/Documents/NavGurukul/piper1-gpl")
SPICOR_DIR = BASE_DIR / "content/IISc_SPICOR_Data/IISc_SPICORProject_English_Female_Spk001_HC"
TRANSCRIPT_JSON = SPICOR_DIR / "IISc_SPICORProject_English_Female_Spk001_HC_Transcripts.json"
WAV_DIR = SPICOR_DIR / "wav"
OUTPUT_DIR = BASE_DIR / "content/dataset_24hr"
CONFIG_JSON = BASE_DIR / "configs/en_US-ljspeech-medium.onnx.json"

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
        return 0.0

def phonemize_text(text: str) -> str:
    """Convert text to phonemes using espeak-ng."""
    try:
        # Use en-us as base
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
        if word not in US_ENGLISH_VOCAB and not word.isdigit():
            non_english.append(word)
    return non_english

def load_dataset() -> Dict[str, Sample]:
    """Load the transcript JSON and create Sample objects."""
    print("Loading transcripts...")
    with open(TRANSCRIPT_JSON, 'r') as f:
        data = json.load(f)
    
    samples = {}
    missing_wavs = 0
    
    transcripts = data.get("Transcripts", {})
    
    for sample_id, info in transcripts.items():
        wav_path = WAV_DIR / f"{sample_id}.wav"
        
        if not wav_path.exists():
            possible_files = list(WAV_DIR.glob(f"{sample_id}*.wav"))
            if possible_files:
                wav_path = possible_files[0]
            else:
                missing_wavs += 1
                continue
        
        transcript_text = info.get('Transcript', '')
        samples[sample_id] = Sample(
            id=sample_id,
            transcript=transcript_text,
            domain=info.get('Domain', 'UNKNOWN'),
            wav_path=wav_path,
            word_count=len(transcript_text.split())
        )
    
    print(f"Loaded {len(samples)} samples ({missing_wavs} missing wav files)")
    return samples

def process_sample(args):
    """Helper for parallel processing."""
    sample_id, sample, target_phonemes = args
    
    # Get duration
    try:
        sample.duration = get_wav_duration(sample.wav_path)
        
        # Get phonemes
        sample.phonemes = phonemize_text(sample.transcript)
        sample.phoneme_set = extract_phoneme_set(sample.phonemes)
        
        if target_phonemes:
            sample.phoneme_set = sample.phoneme_set.intersection(target_phonemes)
        
        # Identify non-English words
        sample.non_english_words = get_non_english_words(sample.transcript)
        return sample_id, sample
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        return sample_id, sample

def analyze_samples(samples: Dict[str, Sample], target_phonemes: Set[str]) -> Dict[str, Sample]:
    """Analyze samples for duration, phonemes, and features using multiprocessing."""
    print("\nAnalyzing samples (using multiprocessing)...")
    
    from multiprocessing import Pool, cpu_count
    
    # Prepare args
    tasks = []
    for sample_id, sample in samples.items():
        tasks.append((sample_id, sample, target_phonemes))
    
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes for {len(tasks)} samples...")
    
    processed_samples = {}
    with Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(process_sample, tasks, chunksize=50)
        
        for i, (sample_id, sample) in enumerate(results):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(tasks)} samples...")
            processed_samples[sample_id] = sample
            
    # Update the original dict (in place update not possible easily with parallel, so we return new)
    # Actually we can just update the objects since we return them
    # But wait, objects across processes are pickled.
    # So we need to put the returned sample back into the dict.
    
    # Print statistics
    total_duration = sum(s.duration for s in processed_samples.values())
    print(f"\nTotal dataset duration: {total_duration/3600:.2f} hours")
    if len(processed_samples) > 0:
        print(f"Average sample duration: {total_duration/len(processed_samples):.2f} seconds")
        
    return processed_samples

def selection_logic(samples: Dict[str, Sample]) -> List[Sample]:
    """
    Select samples based on:
    1. Equal representation for all categories (domains).
    2. Max 60% samples with >= 5 non-English words.
    3. Remaining (at least 40%) samples with < 5 non-English words.
    """
    print("\nSelecting samples with constraints...")
    print(f"Target Word Count: {TARGET_WORD_COUNT}")
    print("Constraints: Equal per domain, <= 60% 'Heavy' Non-English (>= 5 NE words)")
    
    # Group by domain
    samples_by_domain = defaultdict(list)
    for sample in samples.values():
        samples_by_domain[sample.domain].append(sample)
    
    domains = sorted(samples_by_domain.keys())
    print(f"Found {len(domains)} domains: {domains}")
    
    # Calculate targets
    target_words_per_domain = TARGET_WORD_COUNT / len(domains)
    heavy_ne_ratio = 0.6
    target_heavy = target_words_per_domain * heavy_ne_ratio
    target_light = target_words_per_domain * (1 - heavy_ne_ratio)
    
    print(f"Target per domain: {target_words_per_domain:.0f} words")
    print(f"  - Heavy NE (>=5 words, <=60%): {target_heavy:.0f} words")
    print(f"  - Light NE (<5 words, >=40%): {target_light:.0f} words")
    
    selected_ids = set()
    selected_samples = []
    
    total_words = 0
    
    for domain in domains:
        domain_items = samples_by_domain[domain]
        
        # Separate into two buckets
        # Bucket A: Heavy NE (>= 5 non-english words)
        # Bucket B: Light NE (< 5 non-english words)
        heavy_items = []
        light_items = []
        
        for s in domain_items:
            # We don't filter by ratio per sample anymore, just absolute count as per new req
            ne_count = len(s.non_english_words)
            
            if ne_count >= 5:
                heavy_items.append(s)
            else:
                light_items.append(s)
        
        # Sort by quality/features
        # Prioritize high NE count for Heavy bucket only if we really want to maximize it, 
        # but usually we want good phoneme coverage.
        heavy_items.sort(key=lambda s: (len(s.phoneme_set), len(s.non_english_words), -s.duration), reverse=True)
        # For light items, prioritizing those with *some* NE words might help diversity, or just phonemes.
        light_items.sort(key=lambda s: (len(s.phoneme_set), len(s.non_english_words), -s.duration), reverse=True)
        
        domain_word_count = 0
        domain_heavy_count = 0
        domain_light_count = 0
        
        # 1. Fill Heavy Quota (Max 60%)
        current_heavy_words = 0
        for s in heavy_items:
            if current_heavy_words >= target_heavy:
                break
            
            if s.duration < 0.5: continue
            
            selected_samples.append(s)
            selected_ids.add(s.id)
            current_heavy_words += s.word_count
            domain_word_count += s.word_count
            domain_heavy_count += 1
            
        # 2. Fill Light Quota (Min 40% + overflow from Heavy if any)
        # Fill the rest of the domain target with Light items
        
        remaining_domain_target = target_words_per_domain - current_heavy_words
        
        current_light_words = 0
        for s in light_items:
            if (current_heavy_words + current_light_words) >= target_words_per_domain:
                break
                
            if s.duration < 0.5: continue
                
            selected_samples.append(s)
            selected_ids.add(s.id)
            current_light_words += s.word_count
            domain_word_count += s.word_count
            domain_light_count += 1
            
        total_words += domain_word_count
        print(f"  Domain {domain:<15}: {domain_word_count} words (Heavy: {domain_heavy_count}, Light: {domain_light_count})")

    # Phase 2: If we are short, fill with leftovers
    if total_words < TARGET_WORD_COUNT:
        print("\nPhase 2: Filling remaining global capacity...")
        
        leftovers_heavy = []
        leftovers_light = []
        
        for sample in samples.values():
            if sample.id not in selected_ids:
                if len(sample.non_english_words) >= 5:
                    leftovers_heavy.append(sample)
                else:
                    leftovers_light.append(sample)
        
        leftovers_heavy.sort(key=lambda s: (len(s.phoneme_set), -s.duration), reverse=True)
        leftovers_light.sort(key=lambda s: (len(s.phoneme_set), -s.duration), reverse=True)
        
        # Recalculate global heavy ratio
        current_heavy_words = sum(s.word_count for s in selected_samples if len(s.non_english_words) >= 5)
        
        added_count = 0
        while total_words < TARGET_WORD_COUNT:
            can_add_heavy = False
            
            if leftovers_heavy:
                cand = leftovers_heavy[0]
                new_heavy_total = current_heavy_words + cand.word_count
                new_total = total_words + cand.word_count
                if (new_heavy_total / new_total) <= 0.6:
                    can_add_heavy = True
            
            s = None
            if can_add_heavy:
                s = leftovers_heavy.pop(0)
                current_heavy_words += s.word_count
            elif leftovers_light:
                s = leftovers_light.pop(0)
            else:
                break
                
            if s.duration < 0.5: continue
            
            selected_samples.append(s)
            selected_ids.add(s.id)
            total_words += s.word_count
            added_count += 1
            
        print(f"  Added {added_count} extra samples.")
    
    print(f"Selection complete. Selected {len(selected_samples)} samples, {total_words} words.")
    return selected_samples

def sort_samples(selected: List[Sample]) -> List[Sample]:
    """
    Sort samples:
    1. Heavy NE (>= 5 non-English words) first.
    2. Light NE (< 5 non-English words) second.
    Within each group, keep existing sort or sort by ID/Domain if needed?
    We'll assume the input list has some meaningful order we want to preserve *within* the groups,
    or we just simple-split.
    """
    heavy = [s for s in selected if len(s.non_english_words) >= 5]
    light = [s for s in selected if len(s.non_english_words) < 5]
    
    # Optional: deterministic sort within groups?
    # heavy.sort(key=lambda s: s.id) 
    # light.sort(key=lambda s: s.id)
    # The selection logic already sorts by quality. Let's keep that order.
    
    print(f"\nSorting: {len(heavy)} Heavy NE samples first, followed by {len(light)} Light NE samples.")
    return heavy + light

def save_selected_dataset(selected: List[Sample], skip_copy: bool = False) -> None:
    """Save the selected dataset as metadata.csv and optionally copy wav files."""
    import shutil
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_output_dir = OUTPUT_DIR / "wavs"
    wav_output_dir.mkdir(exist_ok=True)
    
    # Write metadata.csv
    metadata_path = OUTPUT_DIR / "metadata.csv"
    with open(metadata_path, 'w') as f:
        for sample in selected:
            # Format: filename|text
            filename = sample.wav_path.stem
            f.write(f"{filename}|{sample.transcript}\n")
    
    print(f"\nWritten metadata to {metadata_path}")
    
    if skip_copy:
        print("Skipping file copy as dataset already exists.")
        return

    # Copy wav files
    print(f"Copying {len(selected)} wav files to {wav_output_dir}...")
    
    count = 0
    total = len(selected)
    for i, sample in enumerate(selected):
        if (i + 1) % 500 == 0:
            print(f"  Copied {i + 1}/{total} files...")
        
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
    heavy_ne_count = 0
    light_ne_count = 0
    
    for s in selected:
        covered.update(s.phoneme_set)
        if len(s.non_english_words) >= 5:
            heavy_ne_count += 1
        else:
            light_ne_count += 1
            
    print(f"Heavy Non-English (>=5 words): {heavy_ne_count} ({heavy_ne_count/len(selected)*100:.1f}%) [Target <= 60%]")
    print(f"Light Non-English (<5 words): {light_ne_count} ({light_ne_count/len(selected)*100:.1f}%) [Target >= 40%]")
    
    if target_phonemes:
        print(f"Phoneme Coverage: {len(covered)}/{len(target_phonemes)} ({len(covered)/len(target_phonemes)*100:.1f}%)")

def load_existing_samples_from_dir(wav_dir: Path) -> Dict[str, Sample]:
    """Load sample objects for existing wav files in the output directory."""
    print(f"Loading existing samples from {wav_dir}...")
    existing_wavs = list(wav_dir.glob("*.wav"))
    existing_ids = {p.stem for p in existing_wavs}
    
    # We still need the transcripts to reconstruct Sample objects correctly
    print(f"Reading transcripts from {TRANSCRIPT_JSON}...")
    with open(TRANSCRIPT_JSON, 'r') as f:
        data = json.load(f)
    transcripts = data.get("Transcripts", {})
    
    samples = {}
    for sample_id in existing_ids:
        if sample_id in transcripts:
            info = transcripts[sample_id]
            transcript_text = info.get('Transcript', '')
            
            # Note: Point wav_path to the SPICOR source just so we can read valid headers/duration/analysis
            # OR we can point to the OUTPUT wav.
            # Pointing to OUTPUT wav is safer if we want to process what is actually there.
            wav_path = wav_dir / f"{sample_id}.wav"
            
            samples[sample_id] = Sample(
                id=sample_id,
                transcript=transcript_text,
                domain=info.get('Domain', 'UNKNOWN'),
                wav_path=wav_path,
                word_count=len(transcript_text.split())
            )
        else:
            print(f"Warning: ID {sample_id} not found in transcript JSON.")
            
    print(f"Reconstructed {len(samples)} samples from existing directory.")
    return samples

def main():
    print("="*60)
    print("PIPER TTS - 24-Hour Dataset Preparation")
    print("="*60)
    
    target_phonemes = load_target_phonemes()
    
    # Check if dataset already exists
    wav_output_dir = OUTPUT_DIR / "wavs"
    existing_wav_count = 0
    if wav_output_dir.exists():
        existing_wav_count = len(list(wav_output_dir.glob("*.wav")))
    
    print(f"Found {existing_wav_count} wav files in {wav_output_dir}")
    
    selected = []
    skip_copy = False
    
    # Requirement: "around 10000". Let's say > 9000 is "around" enough to signal existing dataset.
    if existing_wav_count > 9000:
        print("\n>> Existing dataset detected (> 9000 files). Switching to 'Organize Only' mode.")
        print(">> Will load existing files, analyze, sort, and update metadata.csv.")
        
        # Load samples based on what is in the folder
        samples = load_existing_samples_from_dir(wav_output_dir)
        
        # We need to analyze them to get the Non-English counts for sorting
        samples = analyze_samples(samples, target_phonemes)
        
        selected = list(samples.values())
        skip_copy = True
        
    else:
        print("\n>> No sufficient existing dataset found. Starting full selection process.")
        
        # 1. Load full pool
        samples = load_dataset()
        
        # 2. Filter missing
        samples = {k: v for k, v in samples.items() if v.wav_path.exists()}
        
        # 3. Analyze
        samples = analyze_samples(samples, target_phonemes)
        
        # 4. Selection Logic
        selected = selection_logic(samples)
    
    # Post-selection/Post-load sorting
    selected = sort_samples(selected)
    
    print_stats(selected, target_phonemes)
    
    save_selected_dataset(selected, skip_copy=skip_copy)

if __name__ == "__main__":
    main()
