from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from train import clean_text
import joblib

# Initialize the FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API for predicting sentiment of movie reviews using a pre-trained model.",
    version="1.0.0",
)

# Load the pre-trained model and vectorizer
try:
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    model = None
    vectorizer = None
    raise RuntimeError("Model or vectorizer not found. Please run the training script first.")

# Define the request body model
class Review(BaseModel):
    text: str


@app.post("/predict", tags=["Sentiment Prediction"])
async def predict_sentiment(review: Review):
    if model is None or vectorizer is None:
        raise RuntimeError("Model or vectorizer not loaded. Please check the setup.")
    
    # Clean the input text
    cleaned_text = clean_text(review.text)
    # Vectorize the cleaned text
    vectorized_text = vectorizer.transform([cleaned_text])
    # Predict the sentiment
    prediction = model.predict(vectorized_text)
    prediction_proba = model.predict_proba(vectorized_text)
    # Get the probability of the predicted class
    confidence = prediction_proba[0].max()

    return {
        "review": review.text,
        "sentiment": prediction[0],
        "confidence": float(confidence)
    }
 
# Define the root endpoint

@app.get("/", tags=["Root"], include_in_schema=False)
async def redirect_to():
    return RedirectResponse(url="/docs")
