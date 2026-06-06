# Spam Email Detection

An end-to-end **Machine Learning** project that classifies emails as **Spam** or **Ham** (legitimate) using Natural Language Processing and a Naive Bayes classifier. Features a premium dark-themed web interface built with Flask.

---

## Features

- **Synthetic Dataset Generator** - Creates 1,200+ realistic spam and ham emails for training
- **NLP Text Preprocessing** - Full pipeline: cleaning, tokenization, stopword removal, Porter stemming
- **ML Classification** - TF-IDF vectorization + Multinomial Naive Bayes with high accuracy
- **Beautiful Web UI** - Dark theme with glassmorphism, gradient accents, and smooth animations
- **Confidence Scoring** - Shows prediction confidence as an animated progress bar
- **MongoDB Integration** - Optional database logging for classification results
- **Sample Emails** - One-click buttons to test with pre-filled spam/ham examples

---

## Tech Stack

| Layer          | Technology                        |
| -------------- | --------------------------------- |
| Language       | Python 3                          |
| Web Framework  | Flask                             |
| ML / NLP       | Scikit-learn, NLTK                |
| Data           | Pandas, NumPy                     |
| Vectorization  | TF-IDF (max 5,000 features)      |
| Classifier     | Multinomial Naive Bayes           |
| Database       | MongoDB (optional, via PyMongo)   |
| Frontend       | HTML5, CSS3, Vanilla JS           |
| Serialization  | Joblib                            |

---

## Project Structure

```
spam-email-detection/
├── app.py                     # Flask web application
├── model/
│   ├── train.py               # Model training script
│   ├── predict.py             # Prediction logic
│   ├── classifier.pkl         # Trained model (generated)
│   └── vectorizer.pkl         # TF-IDF vectorizer (generated)
├── data/
│   ├── generate_data.py       # Synthetic email dataset generator
│   └── emails.csv             # Dataset (generated)
├── preprocessing/
│   └── text_processor.py      # Text cleaning & preprocessing
├── database/
│   └── mongo_handler.py       # MongoDB operations (optional)
├── templates/
│   ├── index.html             # Email input form
│   └── result.html            # Classification result page
├── static/
│   └── style.css              # Premium dark theme styling
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Navigate to the project

```bash
cd spam-email-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

## How to Run

### Step 1 - Generate the dataset

```bash
python data/generate_data.py
```

Creates `data/emails.csv` with 1,200 labeled emails (660 ham + 540 spam).

### Step 2 - Train the model

```bash
python model/train.py
```

This will:
- Load and preprocess the dataset
- Apply TF-IDF vectorization (max 5,000 features)
- Train a Multinomial Naive Bayes classifier
- Print accuracy, precision, recall, F1-score, and confusion matrix
- Save `model/classifier.pkl` and `model/vectorizer.pkl`

### Step 3 - Run the web application

```bash
python app.py
```

Open your browser and navigate to **http://127.0.0.1:5001**

---

## Model Performance

| Metric    | Score    |
| --------- | -------- |
| Accuracy  | 100.00%  |
| Precision | 100.00%  |
| Recall    | 100.00%  |
| F1-Score  | 100.00%  |

### Confusion Matrix

|                  | Predicted Ham | Predicted Spam |
|------------------|---------------|----------------|
| **Actual Ham**   | 132 (TN)      | 0 (FP)         |
| **Actual Spam**  | 0 (FN)        | 108 (TP)       |

---

## How It Works

1. **Data Generation** - Synthetic emails are generated using templates with random variations
2. **Preprocessing** - Raw text is cleaned (lowercase, remove punctuation/URLs/numbers), tokenized, stopwords removed, and Porter-stemmed
3. **Feature Extraction** - TF-IDF converts preprocessed text into numerical feature vectors
4. **Classification** - Multinomial Naive Bayes predicts spam/ham probability
5. **Web Interface** - Flask serves the prediction through a premium dark-themed UI

### Text Preprocessing Pipeline

```
Raw Email Text
    -> Lowercase
    -> Remove URLs, emails, HTML tags
    -> Remove numbers & punctuation
    -> Tokenization (NLTK)
    -> Stopword Removal
    -> Porter Stemming
    -> Clean text for vectorization
```

---

## Usage

1. Open the web app at `http://127.0.0.1:5001`
2. Paste an email in the textarea (or click a sample button)
3. Click **Classify Email**
4. View the result - **SPAM** (red) or **HAM** (green) with confidence %

---

## MongoDB (Optional)

The application works **without MongoDB**. If you want to enable database logging:

1. Install and start MongoDB locally (`mongodb://localhost:27017/`)
2. The app will automatically detect and connect to MongoDB
3. All classification results will be logged to the `spam_detection_db` database

---

## License

This project is open-source and available for educational and portfolio purposes.
