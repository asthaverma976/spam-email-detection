"""
Prediction Module
Loads the trained classifier and vectorizer, and provides
a function to classify new email text as spam or ham.
"""

import os
import sys

import joblib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.text_processor import TextProcessor

# ─── Module-level cache ──────────────────────────────────────────────────────
_classifier = None
_vectorizer = None
_processor = None


def _load_model():
    """Load the trained model and vectorizer from disk (cached)."""
    global _classifier, _vectorizer, _processor

    if _classifier is not None and _vectorizer is not None:
        return

    model_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_path = os.path.join(model_dir, "classifier.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    if not os.path.exists(classifier_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "Model files not found. Run 'python model/train.py' first to train the model."
        )

    _classifier = joblib.load(classifier_path)
    _vectorizer = joblib.load(vectorizer_path)
    _processor = TextProcessor()
    print("[Predict] Model and vectorizer loaded successfully.")


def predict_email(email_text: str) -> dict:
    """
    Classify an email as spam or ham.

    Args:
        email_text: Raw email text string.

    Returns:
        dict with keys:
            - prediction: 'spam' or 'ham'
            - confidence: float percentage (0-100)
            - preprocessed_text: the cleaned/stemmed text used for prediction
    """
    _load_model()

    # Preprocess
    preprocessed = _processor.preprocess(email_text)

    # Vectorize
    features = _vectorizer.transform([preprocessed])

    # Predict
    prediction_encoded = _classifier.predict(features)[0]
    probabilities = _classifier.predict_proba(features)[0]

    prediction = "spam" if prediction_encoded == 1 else "ham"
    confidence = max(probabilities) * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "preprocessed_text": preprocessed,
    }


# ─── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_emails = [
        "Congratulations! You've won $1,000,000! Click here to claim your prize NOW!",
        "Hi team, the meeting has been moved to 3 PM. Please update your calendars.",
        "URGENT: Your bank account has been compromised. Click here immediately!",
        "Hey, are we still on for dinner tonight? Let me know!",
        "FREE iPhone giveaway! Limited time offer! Act now before it's too late!",
    ]

    print("=" * 70)
    print("SPAM DETECTION - PREDICTION DEMO")
    print("=" * 70)

    for email in test_emails:
        result = predict_email(email)
        label = result["prediction"].upper()
        conf = result["confidence"]
        icon = "🚫" if label == "SPAM" else "✅"
        print(f"\n{icon} [{label}] ({conf:.1f}% confidence)")
        print(f"   Email: {email[:80]}...")
        print(f"   Processed: {result['preprocessed_text'][:60]}...")
