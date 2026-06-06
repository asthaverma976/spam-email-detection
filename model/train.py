"""
Model Training Script
Trains a Multinomial Naive Bayes classifier on the synthetic email dataset
using TF-IDF features. Saves the trained model and vectorizer for inference.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Add project root to path so we can import preprocessing
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.text_processor import TextProcessor


def train_model():
    """
    Full training pipeline:
    1. Load dataset
    2. Preprocess all email text
    3. Vectorize with TF-IDF
    4. Train Multinomial Naive Bayes
    5. Evaluate and print metrics
    6. Save model artifacts
    """

    # ── 1. Load Data ──────────────────────────────────────────────────────
    data_path = os.path.join(PROJECT_ROOT, "data", "emails.csv")

    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}")
        print("Run 'python data/generate_data.py' first to create the dataset.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} emails ({df['label'].value_counts().to_dict()})")
    print("=" * 60)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print("Preprocessing emails...")
    processor = TextProcessor()
    df["processed_text"] = df["email_text"].apply(processor.preprocess)

    # Convert labels to binary (spam=1, ham=0)
    df["label_encoded"] = df["label"].map({"spam": 1, "ham": 0})

    X = df["processed_text"]
    y = df["label_encoded"]

    # ── 3. Train/Test Split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
    print("=" * 60)

    # ── 4. TF-IDF Vectorization ──────────────────────────────────────────
    print("Vectorizing with TF-IDF (max_features=5000)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print("=" * 60)

    # ── 5. Train Classifier ──────────────────────────────────────────────
    print("Training Multinomial Naive Bayes classifier...")
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train_tfidf, y_train)
    print("Training complete!")
    print("=" * 60)

    # ── 6. Evaluate ──────────────────────────────────────────────────────
    y_pred = classifier.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision : {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({f1 * 100:.2f}%)")
    print("=" * 60)

    print("\nCONFUSION MATRIX")
    print("-" * 30)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN (Ham correct)   : {cm[0][0]}")
    print(f"  FP (Ham -> Spam)   : {cm[0][1]}")
    print(f"  FN (Spam -> Ham)   : {cm[1][0]}")
    print(f"  TP (Spam correct)  : {cm[1][1]}")
    print("-" * 30)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # ── 7. Save Model Artifacts ──────────────────────────────────────────
    model_dir = os.path.dirname(os.path.abspath(__file__))

    classifier_path = os.path.join(model_dir, "classifier.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    joblib.dump(classifier, classifier_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\nModel saved  -> {classifier_path}")
    print(f"Vectorizer saved -> {vectorizer_path}")
    print("\nTraining pipeline complete! [DONE]")


if __name__ == "__main__":
    train_model()
