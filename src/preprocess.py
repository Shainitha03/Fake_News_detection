"""
Text Preprocessing Module for Fake News Detection.

This module performs:
1. Lowercasing
2. Removal of URLs, HTML tags, and special characters / punctuation
3. Removal of English stopwords using NLTK
4. Word stemming using PorterStemmer
5. Dataset loading and merging (Fake.csv + True.csv)
"""

import re
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources automatically if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Initialize Stopwords and Porter Stemmer
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()


def clean_text(text: str) -> str:
    """
    Cleans raw news text using classical NLP preprocessing techniques.
    
    Args:
        text (str): Raw input news text or article.
        
    Returns:
        str: Processed, stemmed, and cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase text
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 4. Remove Reuters prefix patterns common in real news (e.g., "WASHINGTON (Reuters) - ")
    # This prevents the model from relying purely on publisher watermarks.
    text = re.sub(r'^.*?\([a-zA-Z\s]+\)\s*-\s*', ' ', text)

    # 5. Remove numbers, punctuation, and non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 6. Tokenize by whitespace, remove stopwords, and apply stemming
    words = text.split()
    cleaned_words = [
        stemmer.stem(word) 
        for word in words 
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned_words)


def load_and_prepare_dataset(fake_csv_path: str, true_csv_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Loads Fake and True news CSVs, creates binary labels (0 = Fake, 1 = Real),
    combines title + text, and shuffles the dataset.
    
    Args:
        fake_csv_path (str): Path to Fake.csv
        true_csv_path (str): Path to True.csv
        sample_size (int, optional): Optional subset size for faster training.
        
    Returns:
        pd.DataFrame: Combined DataFrame with columns ['content', 'label']
    """
    if not os.path.exists(fake_csv_path):
        raise FileNotFoundError(f"Could not find fake news file at: {fake_csv_path}")
    if not os.path.exists(true_csv_path):
        raise FileNotFoundError(f"Could not find true news file at: {true_csv_path}")

    # Load CSVs
    print("Loading Fake and True news datasets...")
    df_fake = pd.read_csv(fake_csv_path)
    df_true = pd.read_csv(true_csv_path)

    # Assign labels: 0 for Fake news, 1 for Real/True news
    df_fake['label'] = 0
    df_true['label'] = 1

    # If sample size specified, take balanced sample from each
    if sample_size:
        half_sample = sample_size // 2
        df_fake = df_fake.sample(n=min(half_sample, len(df_fake)), random_state=42)
        df_true = df_true.sample(n=min(half_sample, len(df_true)), random_state=42)

    # Combine datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Combine 'title' and 'text' columns into a single 'content' column
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + " " + df['text']

    # Keep only content and label
    df = df[['content', 'label']].dropna()

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset prepared successfully! Total articles: {len(df)} (Fake: {sum(df['label'] == 0)}, Real: {sum(df['label'] == 1)})")
    return df


if __name__ == '__main__':
    # Quick demonstration of the cleaning function
    sample_article = "BREAKING: Scientists have discovered a new planet orbiting Proxima Centauri! Visit https://example.com for details."
    print("Original Text:\n", sample_article)
    print("\nCleaned Text:\n", clean_text(sample_article))
