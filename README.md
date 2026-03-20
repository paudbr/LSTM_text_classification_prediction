## 🌐 [View project page →](https://paudbr.github.io/taylor-swift-nlp/)
# 🎤 Taylor Swift NLP — The Eras Tour of Machine Learning

> *Text Organizing Using Recurrent networks*

**Taylor Swift NLP** is a machine learning project that uses Taylor Swift's complete song lyrics discography as a dataset to explore two core NLP tasks: **album classification** and **song lyric generation**.

Built by **Lucía de Lamadrid** & **Paula de Blas Rioja** — *Aprendizaje Automático (Machine Learning), 2024*

---

## ✨ What does this project do?

Given the complete Taylor Swift discography (11 albums, ~240 songs), we explore:

| Task | Methods | Best Accuracy |
|------|---------|--------------|
| **Album Classification** | Multinomial NB, Bagging, SVM, LSTM | ~66% |
| **Lyric Generation** | LSTM (PyTorch), LSTM (TensorFlow) | — |
| **Interactive Classifier** | Multinomial NB + CLI interface | ~65% |

---

## 📁 Repository Structure

```
taylor-swift-nlp/
│
├── data/
│   ├── songs_modif2.csv          # Main dataset — lyrics with album labels (9 classes)
│   ├── songs_-_songs.csv         # Dataset for LSTM text generation (PyTorch)
│   └── cleaned.parquet           # Pre-cleaned dataset for LSTM generation (TensorFlow)
│
├── notebooks/
│   ├── Taylor_classification_A.ipynb   # Part A: Multinomial NB, all 11 album labels
│   ├── Taylor_classification_B.ipynb   # Part B: Reduced labels, SVM, LSTM classification
│   ├── lstm_pytorch.ipynb              # LSTM lyric generation — PyTorch implementation
│   └── lstm_tensorflow.ipynb           # LSTM lyric generation — TensorFlow implementation
│
├── src/
│   ├── classifier.py             # Multinomial NB classifier (training + prediction)
│   └── run_classifier.py         # Interactive CLI interface for album prediction
│
├── models/                       # Saved model files (generated at runtime)
│   ├── model_classification.pkl
│   └── vectorizer.pkl
│
├── results/
│   └── figures/                  # Confusion matrices, word clouds, training curves
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

The dataset contains the **complete Taylor Swift discography** with song lyrics scraped from public sources. Each row corresponds to one song with its album label.

**Albums included:**

| Label in dataset | Album |
|-----------------|-------|
| `Taylor Swift` | Taylor Swift *(debut, removed in reduced label set)* |
| `Fearless (Taylor's Version)` | Fearless (Taylor's Version) |
| `Speak Now (Taylor's Version)` | Speak Now (Taylor's Version) |
| `Red (Taylor's Version)` | Red (Taylor's Version) |
| `nineteeneightynine (Taylor's Version)` | 1989 (Taylor's Version) |
| `reputation` | reputation |
| `Lover` | Lover |
| `folkmore` | *folklore* + *evermore* (merged — written during pandemic, lyrically similar) |
| `Midnights (3am Edition)` | Midnights (3am Edition) |
| `The Tortured Poets Department` | The Tortured Poets Department |

> **Note on label reduction:** The debut album *Taylor Swift* and the *folklore/evermore* pair were treated specially. The debut was dropped (too few songs), and *folklore* + *evermore* were merged into a single `folkmore` label since Taylor wrote both during the COVID-19 pandemic and they share very similar musical and lyrical styles. This reduced the label space from 11 → 9.

---

## 🧠 Methods

### 1. Text Preprocessing (NLTK)

All lyrics go through a standard NLP preprocessing pipeline:

```
Raw lyrics
    ↓ Lowercase
    ↓ Remove non-alphanumeric characters
    ↓ Tokenize (split into words)
    ↓ Remove stopwords (NLTK English stopword list)
    ↓ Porter Stemmer (reduce words to root form)
    → Clean token sequence
```

**Example:**
```
Input:  "Of those nights We ditch the whole scene It feels like one of those nights We wont be sleepin'"
Output: "night ditch whole scene feel like one night wont sleepin"
```

### 2. Feature Extraction (CountVectorizer)

Processed tokens are converted to numerical vectors using `CountVectorizer` (max 5,000 features). Each lyric chunk is represented as a sparse vector of word counts.

### 3. Lyric Chunking

Full lyrics are split into chunks of **35 words** (shuffled) before training. This significantly increases the number of training samples and prevents the model from memorising whole songs.

### 4. Classification Models

#### Multinomial Naive Bayes (MNB)

Applies Bayes' theorem to compute the probability that a lyric chunk belongs to each album:

$$P(c_i | d_j) = \frac{P(c_i) \cdot P(d_j | c_i)}{P(d_j)}$$

- **Without label reduction:** Accuracy ~63.5%, +Bagging ~64%
- **With label reduction (9 classes):** Accuracy ~64.5%, +Bagging ~66%

Albums with the most songs (Taylor's Version re-recordings) consistently had the best per-class recall, as expected given the larger number of training samples.

#### Support Vector Machine (SVM)

Trained on the same reduced-label dataset. Accuracy ~59% — slightly below MNB for this task.

#### LSTM Classification

Architecture: `Embedding → LSTM → Dropout(0.1) → Dense(softmax)`  
Training accuracy: ~100% (overfitting), Test accuracy: **~64%** — comparable to MNB.

### 5. LSTM Text Generation

Two implementations are provided:

| Implementation | Framework | Data preprocessing | Text quality |
|---|---|---|---|
| `lstm_pytorch.ipynb` | PyTorch | Manual | Good |
| `lstm_tensorflow.ipynb` | TensorFlow/Keras | Keras `Tokenizer` | **Best** |

**Pipeline:**
```
Lyrics → Clean → Tokenize → Numerical encoding
    → Input sequences (sliding window)
    → Padding (pre-pad with zeros)
    → Split: predictors / label (next word)
    → Train/Validation split
    → Embedding → LSTM → Dropout → Dense(softmax)
```

**Training:** 50 epochs (GPU — Google Colab), Adam optimizer, categorical cross-entropy loss.  
**Metrics:** Accuracy + Perplexity  
**Best result:** Accuracy 0.755, Perplexity 2.681

**Generated text example (seed: "The"):**
```
Dreams Are Just Met — tAylorSwIft

Verse 1:
The first time you ever saw me cry
Dreams are you just met me in a crowded room
Out and a girl dress, yeah yeah
Just like bad time is cold hard

Chorus:
Not many golden, only see you
See us when I was dead
I do not know how to be whole life girl
I do not wanna lose you around at midnight
```

---

## 🖥️ Interactive Classifier (CLI)

The interactive classifier lets you paste any song lyric and predicts which Taylor Swift album it most likely belongs to.

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
python src/run_classifier.py
```

**Example session:**
```
Welcome to the album classifier. Type a song lyric and I'll tell you which Taylor Swift album it belongs to.

Please enter the song lyrics (or type 'quit' to exit):
> Salt air, and the rust on your door I never needed anything more
  Whispers of "Are you sure?" "Never have I ever before"
  But I can see us lost in the memory August slipped away...

The entered lyrics belong to the album: folkmore
Is that correct? (Yes/No): Yes
Great! The prediction was correct.
```

The trained model and vectorizer are saved via `joblib` and reloaded on subsequent runs — no retraining needed unless you restart from scratch.

> ⚠️ **Note:** Since training accuracy is ~65%, the classifier will fail on many inputs. This is expected given the high lexical overlap between albums (Taylor Swift reuses many thematic words across her entire discography).

---

## 📊 Results Summary

| Model | Label set | Accuracy | Notes |
|-------|-----------|----------|-------|
| Multinomial NB | 11 classes | 63.5% | Baseline |
| Multinomial NB + Bagging | 11 classes | 64.0% | Marginal improvement |
| Multinomial NB | 9 classes | 64.5% | After label reduction |
| Multinomial NB + Bagging | 9 classes | **66.0%** | Best classical model |
| SVM | 9 classes | 59.0% | Worse than NB |
| LSTM (classification) | 9 classes | 64.0% | Comparable to NB, more complex |
| LSTM (generation) | — | acc=0.755, perp=2.681 | TensorFlow, 50 epochs |

---

## 🔍 Key Findings

**Text preprocessing is critical.** Stopword removal and stemming are essential for reducing noise. Taylor Swift uses many emotional/generic words across all albums ("love", "know", "never", "time") which naturally limits classification accuracy.

**Simple models compete with complex ones.** Multinomial NB achieves nearly identical accuracy to LSTM for this classification task, with far less computational cost.

**Data imbalance matters.** Albums with more songs (re-recorded Taylor's Versions include vault tracks) consistently show higher per-class recall. The debut album performed poorly due to having very few songs.

**Label engineering improves performance.** Merging lyrically similar albums (*folklore* + *evermore* → `folkmore`) and removing the debut improved overall accuracy.

**Potential improvement:** Treating high-frequency cross-album words as additional stop words could further improve discrimination between albums — at the cost of some information.

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
nltk
joblib
torch
tensorflow
keras
wordcloud
matplotlib
pyarrow
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

You will also need to download NLTK data the first time:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 📓 Notebooks Guide

| Notebook | Description |
|----------|-------------|
| `Taylor_classification_A.ipynb` | EDA, word clouds per album, Multinomial NB with all 11 labels, confusion matrix, bagging |
| `Taylor_classification_B.ipynb` | Reduced label set (9 classes), SVM, LSTM classification, comparison |
| `lstm_pytorch.ipynb` | Lyric generation with PyTorch LSTM — manual preprocessing |
| `lstm_tensorflow.ipynb` | Lyric generation with TensorFlow/Keras — best text quality output |

---

## 🎯 Future Work

- Add artist-specific stop words (recurring words across all albums)
- Try TF-IDF instead of raw counts
- Experiment with pre-trained embeddings (GloVe, Word2Vec) for classification
- Fine-tune a small language model for generation (GPT-2)
- More training epochs for LSTM generation (limited by hardware)
- Web interface for the interactive classifier

---

## 👩‍💻 Authors

**Lucía de Lamadrid** & **Paula de Blas Rioja**  
Machine Learning course — 2024

---

## 📄 License

This project is for educational purposes only. All song lyrics belong to their respective copyright holders (Taylor Swift / Republic Records / Big Machine Records).
