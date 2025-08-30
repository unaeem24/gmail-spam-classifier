import joblib

# Load model and vectorizer
model = joblib.load("best_email_classifier.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Example: Replace this with your email text
new_email = """
Congratulations! You've won a $1000 gift card. Click here to claim your prize: http://spamlink.com
"""

# If you have a preprocessor, use it here (optional)
# from preprocess import EmailPreprocessor
# preprocessor = EmailPreprocessor()
# new_email = preprocessor.full_preprocess(new_email)

# Vectorize
X_new = vectorizer.transform([new_email])

# Predict
prediction = model.predict(X_new)[0]
prob = model.predict_proba(X_new)[0]

print(f"Prediction: {prediction}")
print(f"Probability (not spam, spam): {prob}")