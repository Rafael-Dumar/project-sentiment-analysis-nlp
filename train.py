
# 1. IMPORTS & SETUP

import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 2. NLTK SETUP
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

# 4. MAIN TRAINING SCRIPT
def main():
    
    print("Starting training process...")

    # Load dataset
    print("Loading IMDB dataset...")
    df = pd.read_csv("IMDB Dataset.csv")

    # Clean text data
    print("Cleaning text data...")
    df['review_cleaned'] = df['review'].apply(clean_text)

    # Feature Engineering (TF-IDF Vectorization)
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['review_cleaned'])
    y = df['sentiment']

    # Model Training
    print("Training the Logistic Regression model on the full dataset...")
    # random_state is used for reproducibility
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    # Save artifacts
    print("Saving vectorizer and model to disk...")
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(model, 'model.pkl')

    print("\n✅ Training complete! Artifacts 'vectorizer.pkl' and 'model.pkl' have been saved.")

if __name__ == '__main__':
    main()