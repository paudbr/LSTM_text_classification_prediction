"""
run_classifier.py
-----------------
Interactive CLI interface for the Taylor Swift album classifier.

Run this script to enter song lyrics and receive a prediction of which
Taylor Swift album they most likely belong to.

The trained model is loaded from disk if it exists; otherwise it is
trained from scratch on the first run and saved automatically.

Usage:
    python src/run_classifier.py

Authors: Lucía de Lamadrid & Paula de Blas Rioja — 2024
"""

import joblib
from classifier import train_and_predict

MODEL_PATH = "models/model_classification.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        🎤  Taylor Swift Album Classifier  🎤             ║
║                                                          ║
║  Enter a song lyric and I'll predict which album         ║
║  it belongs to. Type 'quit' to exit.                     ║
╚══════════════════════════════════════════════════════════╝
"""


def load_model():
    """Attempt to load a pre-trained model and vectorizer from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✓  Pre-trained model loaded successfully.")
        return model, vectorizer
    except FileNotFoundError:
        print("ℹ  No pre-trained model found. Training from scratch on first prediction...")
        return None, None


def main():
    print(BANNER)
    model, vectorizer = load_model()

    while True:
        print()
        lyric = input("Enter song lyrics (or 'quit' to exit):\n> ").strip()

        if lyric.lower() in ("quit", "exit", "q"):
            print("\nThanks for using the Taylor Swift Album Classifier. Goodbye! ✨")
            break

        if not lyric:
            print("Please enter some lyrics.")
            continue

        try:
            predicted_album = train_and_predict(lyric, model, vectorizer)
            print(f"\n🎵  Predicted album:  {predicted_album}\n")

            feedback = input("Was that correct? (yes/no): ").strip().lower()
            if feedback in ("yes", "y"):
                print("✅  Great! The prediction was correct.")
            elif feedback in ("no", "n"):
                print("❌  Sorry about that! The classifier has ~65% accuracy overall,")
                print("    so some misclassifications are expected.")
            else:
                print("Please answer 'yes' or 'no'.")

        except Exception as e:
            print(f"⚠  An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()
