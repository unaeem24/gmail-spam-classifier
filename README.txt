Gmail Spam Classifier

This project uses the Gmail API to fetch emails, preprocesses and vectorizes their content, trains multiple machine learning models to classify emails as **spam** or **not spam**, and allows you to make predictions on new emails.

---

Features

- Fetches emails from your Gmail account using the Gmail API
- Preprocesses email text (HTML removal, stopwords, etc.)
- Trains and evaluates several classifiers (Random Forest, Naive Bayes, SVM, Logistic Regression)
- Saves the best model and vectorizer for future use
- Predicts whether a new email is spam or not

---

Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages include:
- scikit-learn
- google-api-python-client
- google-auth-httplib2
- google-auth-oauthlib
- nltk
- beautifulsoup4
- joblib

3. Gmail API Credentials

- Download your `credentials.json` from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials).
- Place it in the project root directory.


Usage

1. Train the Model

Run the training pipeline to fetch emails, preprocess, vectorize, train, and save the model:

```bash
python gmail_train_pipeline.py
```

This will save:
- `best_email_classifier.joblib` (the trained model)
- `tfidf_vectorizer.joblib` (the fitted vectorizer)

2. Predict on New Emails

Edit `predict_email.py` and replace the `new_email` variable with your email text. Then run:

```bash
python predict_email.py
```

You’ll see output like:

```
Prediction: spam
Probability (not spam, spam): [0.02 0.98]
```

---

Preprocessing

If you want to use advanced preprocessing, edit `predict_email.py` and uncomment the relevant lines to use the `EmailPreprocessor` from `preprocess.py`.

---

File Overview

- `gmail_train_pipeline.py` — Main script for fetching, training, and saving the model/vectorizer
- `train_from_sample.py` — Contains the `EmailClassifierTrainer` class
- `preprocess.py` — Email text preprocessing utilities
- `predict_email.py` — Script for making predictions on new emails

---

Notes

- The first time you run the training script, a browser window will open for Gmail authentication.
- You can adjust the number of emails fetched or the model parameters in the scripts as needed.

---
