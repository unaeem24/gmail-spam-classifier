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