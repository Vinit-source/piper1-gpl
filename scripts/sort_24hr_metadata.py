#!/usr/bin/env python3
"""
Sort the dataset_24hr metadata.csv based on non-English word count in descending order.
"""

import string
from pathlib import Path

# Try to import nltk
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
    print("Warning: NLTK not found. Sorting will fall back to simple length (not accurate).")
    US_ENGLISH_VOCAB = set()
    NLTK_AVAILABLE = False

BASE_DIR = Path("/Users/vinitgore/Documents/NavGurukul/piper1-gpl")
METADATA_PATH = BASE_DIR / "content/dataset_24hr/metadata.csv"

def clean_and_tokenize(text: str) -> list[str]:
    """Converts text to 'no punctuation no case' and splits into words."""
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()
    return clean_text.split()

def get_non_english_count(text: str) -> int:
    """Returns the count of non-English words."""
    if not NLTK_AVAILABLE:
        return 0
        
    tokens = clean_and_tokenize(text)
    count = 0
    for word in tokens:
        if word not in US_ENGLISH_VOCAB and not word.isdigit():
            count += 1
    return count

def main():
    if not METADATA_PATH.exists():
        print(f"Error: File not found at {METADATA_PATH}")
        return

    print(f"Reading metadata from {METADATA_PATH}...")
    
    lines = []
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Found {len(lines)} entries. Calculating non-English counts...")
    
    # Store tuples of (count, line_content, filename)
    # Parsing line: filename|transcript
    processed_entries = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split('|', 1)
        if len(parts) != 2:
            print(f"Warning: Skipping invalid line: {line}")
            continue
            
        filename, transcript = parts
        count = get_non_english_count(transcript)
        processed_entries.append((count, line, filename))
        
    # Sort: Primary = count (ascending), Secondary = filename (ascending)
    processed_entries.sort(key=lambda x: (x[0], x[2]))
    
    print("Sorting complete. Writing back to file...")
    
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        for item in processed_entries:
            f.write(item[1] + "\n")
            
    # Stats
    print("\n" + "="*40)
    print("SORTING STATISTICS")
    print("="*40)
    if processed_entries:
        highest = processed_entries[-1][0]
        lowest = processed_entries[0][0]
        print(f"Highest Non-English Count: {highest}")
        print(f"Lowest Non-English Count:  {lowest}")
        
        # Count buckets
        gte_5 = sum(1 for x in processed_entries if x[0] >= 5)
        lt_5 = len(processed_entries) - gte_5
        print(f"Samples with >= 5 NE words: {gte_5} ({gte_5/len(processed_entries)*100:.1f}%)")
        print(f"Samples with < 5 NE words:  {lt_5} ({lt_5/len(processed_entries)*100:.1f}%)")

    print(f"\nDone. Updated {METADATA_PATH}")

if __name__ == "__main__":
    main()
