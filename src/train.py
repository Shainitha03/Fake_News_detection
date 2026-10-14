"""
Model Training Module for Fake News Detection.

Steps performed:
1. Load dataset (Fake.csv + True.csv) using preprocess module.
2. Clean all text using NLTK preprocessing (lowercasing, stopword removal, stemming).
3. Split into Train (80%) and Test (20%) datasets.
4. Extract features using TF-IDF Vectorizer (Unigrams + Bigrams).
5. Train Logistic Regression Classifier.
6. Evaluate Model (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
7. Save trained model, vectorizer, and evaluation metrics to 'models/' folder.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from preprocess import load_and_prepare_dataset, clean_text


def train_fake_news_model(
    fake_csv: str = "data/Fake.csv",
    true_csv: str = "data/True.csv",
    models_dir: str = "models",
    sample_size: int = None,
    max_features: int = 50000
):
    """
    Main training pipeline.
    """
    start_time = time.time()
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("ACADEMIC NLP PROJECT: FAKE NEWS DETECTION TRAINING")
    print("=" * 60)

    # 1. Load Data
    print("\n[Step 1/5] Loading and merging datasets...")
    df = load_and_prepare_dataset(fake_csv, true_csv, sample_size=sample_size)

    # 2. Text Preprocessing
    print(f"\n[Step 2/5] Cleaning and preprocessing {len(df)} articles (NLTK Tokenization, Stopwords, Stemming)...")
    df['cleaned_content'] = df['content'].apply(clean_text)

    # Filter out any rows that became empty after cleaning
    df = df[df['cleaned_content'].str.strip() != '']
    print(f"-> Preprocessing complete. Usable articles: {len(df)}")

    # 3. Train/Test Split
    print("\n[Step 3/5] Splitting data into Training (80%) and Testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_content'], 
        df['label'], 
        test_size=0.20, 
        random_state=42, 
        stratify=df['label']
    )
    print(f"-> Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 4. TF-IDF Feature Extraction
    print(f"\n[Step 4/5] Extracting TF-IDF features (max_features={max_features}, ngram_range=(1, 2))...")
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"-> TF-IDF Vocabulary Size: {len(tfidf.vocabulary_)} features")

    # 5. Model Training (Logistic Regression)
    print("\n[Step 5/5] Training Logistic Regression Classifier...")
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    print("                 Predicted Fake (0) | Predicted Real (1)")
    print(f"Actual Fake (0): {cm[0][0]:^18} | {cm[0][1]:^18}")
    print(f"Actual Real (1): {cm[1][0]:^18} | {cm[1][1]:^18}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)']))

    # Save artifacts
    model_path = os.path.join(models_dir, "model.pkl")
    tfidf_path = os.path.join(models_dir, "vectorizer.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")

    print("Saving model and vectorizer...")
    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)

    metrics = {
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocab_size": len(tfidf.vocabulary_)
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    total_time = round(time.time() - start_time, 2)
    print(f"\n[DONE] Training and evaluation completed in {total_time} seconds!")
    print(f"Artifacts saved:")
    print(f"  - Model      : {model_path}")
    print(f"  - Vectorizer : {tfidf_path}")
    print(f"  - Metrics    : {metrics_path}")


if __name__ == '__main__':
    train_fake_news_model(sample_size=10000)

