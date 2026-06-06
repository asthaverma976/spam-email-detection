"""
Text Processor Module
Handles all text cleaning and preprocessing for the spam detection pipeline.
Uses NLTK for tokenization, stopwords, and stemming.
"""

import os
import re
import string

import nltk

# ─── Use local NLTK data bundled with the project ────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NLTK_DATA_DIR = os.path.join(_PROJECT_ROOT, "nltk_data")

if os.path.exists(_NLTK_DATA_DIR):
    nltk.data.path.insert(0, _NLTK_DATA_DIR)

# Fallback: download if not found locally
_NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords"]
for _resource in _NLTK_RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{_resource}" if "punkt" in _resource else f"corpora/{_resource}")
    except LookupError:
        try:
            nltk.download(_resource, quiet=True)
        except Exception:
            pass

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class TextProcessor:
    """
    A complete text preprocessing pipeline for NLP tasks.

    Methods
    -------
    clean_text(text)
        Lowercase, strip punctuation, numbers, and extra whitespace.
    tokenize(text)
        Split cleaned text into individual word tokens.
    remove_stopwords(tokens)
        Filter out common English stopwords.
    stem(tokens)
        Apply Porter stemming to reduce words to their root forms.
    preprocess(text)
        Run the full pipeline and return a single cleaned string.
    """

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    # ── Step 1: Clean raw text ────────────────────────────────────────────
    def clean_text(self, text: str) -> str:
        """
        Convert to lowercase, remove punctuation, numbers, URLs,
        email addresses, and collapse extra whitespace.
        """
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ── Step 2: Tokenize ──────────────────────────────────────────────────
    def tokenize(self, text: str) -> list:
        """Split text into individual word tokens using NLTK."""
        return word_tokenize(text)

    # ── Step 3: Remove stopwords ──────────────────────────────────────────
    def remove_stopwords(self, tokens: list) -> list:
        """Remove common English stopwords from the token list."""
        return [token for token in tokens if token not in self.stop_words]

    # ── Step 4: Stem ──────────────────────────────────────────────────────
    def stem(self, tokens: list) -> list:
        """Apply Porter stemming to each token."""
        return [self.stemmer.stem(token) for token in tokens]

    # ── Full Pipeline ─────────────────────────────────────────────────────
    def preprocess(self, text: str) -> str:
        """
        Run the complete preprocessing pipeline:
        clean → tokenize → remove stopwords → stem → rejoin.

        Returns a single preprocessed string ready for vectorization.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        return " ".join(tokens)


# ─── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    processor = TextProcessor()

    sample_spam = (
        "CONGRATULATIONS! You've WON $1,000,000!!! "
        "Click here http://scam.com to CLAIM your prize NOW!!!"
    )
    sample_ham = (
        "Hi team, just a reminder about tomorrow's meeting at 10 AM. "
        "Please prepare your status updates."
    )

    print("=" * 60)
    print("SPAM EXAMPLE")
    print("=" * 60)
    print(f"Original : {sample_spam}")
    print(f"Cleaned  : {processor.clean_text(sample_spam)}")
    print(f"Processed: {processor.preprocess(sample_spam)}")

    print()

    print("=" * 60)
    print("HAM EXAMPLE")
    print("=" * 60)
    print(f"Original : {sample_ham}")
    print(f"Cleaned  : {processor.clean_text(sample_ham)}")
    print(f"Processed: {processor.preprocess(sample_ham)}")
