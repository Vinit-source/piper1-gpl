# Function to convert messages into no punctuation no case text (as per user's request)
import string

def normalize_text(text):
    """Converts text to lowercase and removes all punctuation."""
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    # The split/join ensures multiple spaces are handled cleanly, though not strictly required
    return ' '.join(clean_text.split()).lower()

# --- Main Solution ---

# Requires: pip install pyenchant
import enchant

# 1. Initialize the dictionary ONCE outside the function for efficiency
# We will use this global/module-level variable for quick lookups
try:
    US_ENGLISH_DICT = enchant.Dict("en_US")
    # This set will be used for even faster O(1) lookups after normalization
    US_ENGLISH_SET = set(US_ENGLISH_DICT.suggest(word)[0].lower() for word in US_ENGLISH_DICT.suggest("a")) # Fallback/Initial size estimate
    # Note: enchant doesn't expose the full word list easily, so we rely on check()
except enchant.errors.DictNotFoundError:
    print("Warning: 'en_US' dictionary not found. Please ensure the appropriate dictionary is installed (e.g., using 'sudo apt-get install myspell-en-us' or similar depending on your OS).")
    # Fallback dictionary if US_ENGLISH is not installed
    US_ENGLISH_DICT = None


def process_transcript_for_non_english(transcript_text):
    """
    Checks a transcript for non-US English words, leveraging pyenchant.

    Args:
        transcript_text (str): The text transcript to analyze.

    Returns:
        list: A list of words identified as non-English or not found in the dictionary.
    """
    if US_ENGLISH_DICT is None:
        print("Error: Dictionary not initialized. Cannot perform check.")
        return []

    # 1. Normalize the text (remove punctuation and convert to lowercase)
    normalized_text = normalize_text(transcript_text)

    # 2. Tokenize and check each word
    non_english_words = []
    
    # Check if the word is in the dictionary (case-insensitive check by pyenchant)
    for word in normalized_text.split():
        # A simple check using dictionary.check()
        # This is the most reliable way to use pyenchant for this purpose
        if not US_ENGLISH_DICT.check(word):
            # Check if it's a common number string which check() might fail on
            if not word.isdigit():
                non_english_words.append(word)

    return non_english_words

# --- Example Usage ---

# 1. Your example input JSON structure (simulated)
input_data = {
    "IISc_SPICORProject_EN_F_WEAT_6359": {
        "Transcript": "At the heart of China's space silk road is the BeiDou satellite navigation system",
        "Domain": "WEATHER"
    },
    "Example_With_Non_English": {
        "Transcript": "The cuisine was magnifique, but I still prefer schmaltz in my soup.",
        "Domain": "FOOD"
    },
    "Example_All_English": {
        "Transcript": "This is a standard English sentence.",
        "Domain": "GENERAL"
    }
}

print("--- Processing Transcripts ---")

# Iterate over the simulated JSON structure
for key, data in input_data.items():
    transcript = data["Transcript"]
    
    # Call the function to find non-English words
    non_english_list = process_transcript_for_non_english(transcript)
    
    # Check if the condition (at least one non-English word) is met
    if non_english_list:
        print(f"\n{key} - Non-English Words Found ({len(non_english_list)}):")
        print(f"  Input Transcript: '{transcript}'")
        print(f"  Non-English Words: {non_english_list}")
    else:
        print(f"\n{key} - No non-English words found.")