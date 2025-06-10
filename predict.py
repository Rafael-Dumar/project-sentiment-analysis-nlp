
# 1. IMPORTS & SETUP
import joblib
import re
import nltk
from nltk.corpus import stopwords

# 2. CONSTANTS & HELPER FUNCTIONS
# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    # 1. Remove HTML tags
    text_cleaned = re.sub(r'<.*?>', '', text)
    # 2. Remove special characters and numbers
    text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text_cleaned)
    # 3. Convert to lowercase
    text_cleaned = text_cleaned.lower()
    # 4. Remove extra whitespace
    text_cleaned = ' '.join(text_cleaned.split())
    # 5. Remove stop words
    text_cleaned = ' '.join(word for word in text_cleaned.split() if word not in stop_words)

    return text_cleaned

# 4. MAIN PREDICTION SCRIPT
def predict_sentiment(text):
    try:
        # Load the saved artifacts
        vectorizer = joblib.load('vectorizer.pkl')
        model = joblib.load('model.pkl')
    except FileNotFoundError:
        return "Error: Model or vectorizer not found. Please run train.py first."

    # Clean the input text
    cleaned_text = clean_text(text)

    # Vectorize the cleaned text using the loaded vectorizer
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict the sentiment using the loaded model
    prediction = model.predict(vectorized_text)
    prediction_proba = model.predict_proba(vectorized_text)

    # Get the probability of the predicted class
    confidence = prediction_proba[0].max()
    
    return prediction[0], confidence

# This block runs only when the script is executed directly
if __name__ == '__main__':
    # Example reviews
    positive_review = "This is one of the best movies I have ever seen. The acting was superb, the plot was engaging, and I would recommend it to anyone."
    negative_review = "I loved the way this movie was made. they were capable of making a movie that was so bad that it was good. The acting was terrible, the plot was nonsensical, and I would not recommend it to anyone."
    
    print("--- Sentiment Prediction Tool ---")

    # Predict for the positive review
    sentiment, confidence = predict_sentiment(positive_review)
    print(f"\nReview: '{positive_review}'")
    print(f"--> Predicted Sentiment: {sentiment.upper()} (Confidence: {confidence:.2%})")

    # Predict for the negative review
    sentiment, confidence = predict_sentiment(negative_review)
    print(f"\nReview: '{negative_review}'")
    print(f"--> Predicted Sentiment: {sentiment.upper()} (Confidence: {confidence:.2%})")