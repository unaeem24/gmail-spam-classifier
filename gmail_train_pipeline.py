<<<<<<< HEAD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from train_from_sample import EmailClassifierTrainer
import joblib

# 1. Load Kaggle dataset
df = pd.read_csv('spam_data\combined_data.csv')  # Update path/filename as needed

# 2. Prepare features and labels
X = df['text']   # or the correct column name for email text
y = df['label']  # or the correct column name for label (e.g., 'spam', 'not spam')

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train and evaluate
trainer = EmailClassifierTrainer()
trainer.train(X_train_vec, y_train)
trainer.evaluate(trainer.best_model, X_test_vec, y_test, class_names=["not spam", "spam"])

# 6. Save the trained model and vectorizer
trainer.save_model(trainer.best_model, "best_email_classifier.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
print("Vectorizer saved to tfidf_vectorizer.joblib")
=======
import os
import pickle
import base64
from email import message_from_bytes
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from train_from_sample import EmailClassifierTrainer

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    return service

def get_emails(service, label_ids, max_results=100):
    results = service.users().messages().list(userId='me', labelIds=label_ids, maxResults=max_results).execute()
    messages = results.get('messages', [])
    emails = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
        msg_str = base64.urlsafe_b64decode(msg_data['raw'].encode('ASCII'))
        mime_msg = message_from_bytes(msg_str)
        # Try to extract the plain text part
        if mime_msg.is_multipart():
            for part in mime_msg.walk():
                if part.get_content_type() == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        emails.append(payload.decode(errors='ignore'))
                        break
            else:
                emails.append("")  # No plain text part found
        else:
            payload = mime_msg.get_payload(decode=True)
            if payload:
                emails.append(payload.decode(errors='ignore'))
            else:
                emails.append("")
    return emails

if __name__ == "__main__":
    service = get_gmail_service()
    print("Fetching spam emails...")
    spam_emails = get_emails(service, ['SPAM'], max_results=100)
    print("Fetching not spam emails...")
    not_spam_emails = get_emails(service, ['INBOX'], max_results=100)
    X = spam_emails + not_spam_emails
    y = ['spam'] * len(spam_emails) + ['not spam'] * len(not_spam_emails)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train and evaluate
    trainer = EmailClassifierTrainer()
    trainer.train(X_train_vec, y_train)
    trainer.evaluate(trainer.best_model, X_test_vec, y_test, class_names=["not spam", "spam"])

    # Save the trained model
    trainer.save_model(trainer.best_model, "best_email_classifier.joblib")

    # Save the vectorizer
    import joblib
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    print("Vectorizer saved to tfidf_vectorizer.joblib")
>>>>>>> e716f77073d2f53e33f5f733a04aba61b70601a3
