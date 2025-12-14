import json
import string
import nltk

# --- Step 1: Setup NLTK Dictionary ---
# We check if the dictionary is already downloaded to avoid re-downloading every run
try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading NLTK words corpus...")
    nltk.download('words')

from nltk.corpus import words

# Load the dictionary into a set for O(1) fast lookups.
# normalize to lowercase to match our input processing.
US_ENGLISH_VOCAB = set(w.lower() for w in words.words())


# --- Step 2: Helper Functions ---

def clean_and_tokenize(text):
    """
    Converts text to 'no punctuation no case' and splits into words.
    """
    # Create a translator to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # 1. Remove punctuation
    # 2. Convert to lower case
    clean_text = text.translate(translator).lower()
    
    # Split into a list of words
    return clean_text.split()

def get_non_english_words(text):
    """
    Returns a list of words from the text that are NOT in the English dictionary.
    """
    tokens = clean_and_tokenize(text)
    non_english = []
    
    for word in tokens:
        # Check if word is in dictionary
        # Also ignore pure numbers (e.g., "2024" shouldn't be flagged as non-English)
        if word not in US_ENGLISH_VOCAB and not word.isdigit():
            non_english.append(word)
            
    return non_english


# --- Step 3: Main Processing Logic ---

# Example JSON input (Dictionary of objects)
input_data = {
    "IISc_SPICORProject_EN_F_WEAT_6359": {
        "Transcript": "At the heart of China's space silk road is the BeiDou satellite navigation system",
        "Domain": "WEATHER"
    },
    "Test_Case_Normal": {
        "Transcript": "The weather today is sunny and bright.",
        "Domain": "GENERAL"
    },
    "Test_Case_Foreign": {
        "Transcript": "He shouted namaste to the crowd.",
        "Domain": "CULTURE"
    }
}

print(f"{'ID':<35} | {'Non-English Words'}")
print("-" * 60)

# Loop through each item in the JSON object
for key, value in input_data.items():
    transcript = value.get("Transcript", "")
    
    # Extract non-English words
    unknown_words = get_non_english_words(transcript)
    
    # REQUIREMENT: Accept/Process only if there is at least one non-English word
    if len(unknown_words) > 0:
        print(f"{key:<35} | {unknown_words}")