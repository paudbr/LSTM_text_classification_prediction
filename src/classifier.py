"""
classifier.py
-------------
Multinomial Naive Bayes classifier for Taylor Swift album prediction.

Given a lyric snippet, predicts which Taylor Swift album it most likely
belongs to based on word frequencies.

Dataset: songs_modif2.csv (9-class reduced label set)
  - folklore + evermore merged into 'folkmore'
  - Debut album 'Taylor Swift' excluded

Usage (via run_classifier.py):
    from src.classifier import train_and_predict
    album = train_and_predict(lyric_text)

Authors: Lucía de Lamadrid & Paula de Blas Rioja — 2024
"""

import random
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH = "data/songs_modif2.csv"
MODEL_PATH = "models/model_classification.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
CHUNK_SIZE = 35          # words per lyric chunk
MAX_FEATURES = 5000      # vocabulary size for CountVectorizer

# Albums included in the reduced label set (9 classes)
INCLUDED_ALBUMS = [
    "folkmore",
    "The Tortured Poets Department",
    "Speak Now (Taylor's Version)",
    "Red (Taylor's Version)",
    "Midnights (3am Edition)",
    "Lover",
    "Fearless (Taylor's Version)",
    "reputation",
    "nineteeneightynine (Taylor's Version)",
]


# ── Preprocessing ──────────────────────────────────────────────────────────────

def chunk_lyrics(lyrics: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split a full lyric string into overlapping word chunks of `chunk_size`.

    Each line is split independently to avoid cross-line chunks. The resulting
    chunks are shuffled to prevent ordering bias during training.

    Args:
        lyrics: Full song lyrics as a single string (newline-separated verses).
        chunk_size: Number of words per chunk.

    Returns:
        List of shuffled word-chunk strings.
    """
    lines = lyrics.splitlines()
    chunks = []
    for line in lines:
        words = line.split()
        start = 0
        while start < len(words):
            chunk = " ".join(words[start : start + chunk_size])
            if chunk:
                chunks.append(chunk)
            start += chunk_size
    random.shuffle(chunks)
    return chunks


def preprocess(text: str) -> str:
    """
    Apply the full NLP preprocessing pipeline to a text string:
        1. Lowercase
        2. Remove non-alphanumeric characters
        3. Tokenize
        4. Remove English stopwords
        5. Apply Porter stemmer

    Args:
        text: Raw lyric string.

    Returns:
        Preprocessed string of stemmed tokens joined by spaces.
    """
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    clean = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    tokens = word_tokenize(clean)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


# ── Training & Prediction ──────────────────────────────────────────────────────

def train_and_predict(
    lyric: str,
    model: MultinomialNB | None = None,
    vectorizer: CountVectorizer | None = None,
) -> str:
    """
    Train (or reuse) a Multinomial NB classifier and predict the album for
    the given lyric snippet.

    The trained model and vectorizer are saved to disk after each call so
    they can be reloaded without retraining on subsequent runs.

    Args:
        lyric: Raw lyric text to classify.
        model: Pre-trained MultinomialNB instance, or None to train from scratch.
        vectorizer: Pre-fitted CountVectorizer instance, or None to fit from scratch.

    Returns:
        Predicted album name as a string.
    """
    # ── Load and prepare dataset ──────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, header=0)

    # Merge folklore + evermore into a single label
    df["Album"] = df["Album"].replace({"folklore": "folkmore", "evermore": "folkmore"})

    # Expand each song into multiple lyric chunks
    df = df.assign(
        Lyrics=df["Lyrics"].apply(chunk_lyrics)
    ).explode("Lyrics").reset_index(drop=True)

    # Filter to the 9 target albums
    df = df[df["Album"].isin(INCLUDED_ALBUMS)]

    # ── Vectorize ─────────────────────────────────────────────────────────────
    if vectorizer is None:
        vectorizer = CountVectorizer(max_features=MAX_FEATURES)
        vectorizer.fit(df["Lyrics"])

    X = vectorizer.transform(df["Lyrics"]).toarray()
    y = df["Album"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # ── Train ─────────────────────────────────────────────────────────────────
    if model is None:
        model = MultinomialNB()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    # ── Predict ───────────────────────────────────────────────────────────────
    processed_lyric = preprocess(lyric)
    lyric_vector = vectorizer.transform([processed_lyric]).toarray()
    prediction = model.predict(lyric_vector)

    # Map numeric prediction back to album name
    album_names = df["Album"].unique()
    label_to_album = {i: album for i, album in enumerate(album_names)}
    predicted_album = label_to_album[prediction[0]]

    # ── Save model and vectorizer ─────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return predicted_album
