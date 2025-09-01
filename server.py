from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load your trained model (update the path as needed)
model = joblib.load("best_email_classifier.joblib")  # Make sure model.pkl is in the same directory or provide full path


class EmailRequest(BaseModel):
    email_text: str

@app.post("/predict")
def predict_email(request: EmailRequest):
    email_text = request.email_text
    if not email_text:
        raise HTTPException(status_code=400, detail="Email text is required.")

    # Preprocess the email_text as your model expects
    # For example, if you used a vectorizer:
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    features = vectorizer.transform([email_text])
    prediction = model.predict(features)
    prediction_value = int(prediction[0])

    return {"prediction": prediction_value}

# To run: uvicorn server:app --reload