import joblib

# Load model and vectorizer
model = joblib.load("best_email_classifier.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Example: Replace this with your email text
new_email = """
<<<<<<< HEAD
From: "University Registrar" <registrar@university.edu>
To: you@example.com
Subject: Upcoming Exam Schedule

Dear Student,

Please find attached the exam schedule for the Fall semester. Make sure to review the dates and prepare accordingly.

If you have any questions, feel free to contact the registrar's office.

Best regards,
University Registrar


=======
Congratulations! You've won a $1000 gift card. Click here to claim your prize: http://spamlink.com
>>>>>>> e716f77073d2f53e33f5f733a04aba61b70601a3
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