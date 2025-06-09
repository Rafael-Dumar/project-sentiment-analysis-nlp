# project-sentiment-analysis-nlp
Movie Review Sentiment Classifier using Scikit-learn and NLTK

# NLP Sentiment Analysis for Movie Reviews

![Project Status](https://img.shields.io/badge/status-completed-brightgreen)

## 📖 Project Description
This project implements a complete Machine Learning pipeline to perform Sentiment Analysis on a dataset of 50,000 movie reviews from IMDb. The goal is to classify a review as "positive" or "negative" using Natural Language Processing (NLP) techniques and to compare the performance of different classification algorithms.

## 🚀 Technologies Used
- **Python 3**
- **Pandas:** For data manipulation and analysis.
- **NLTK (Natural Language Toolkit):** For text preprocessing, including tokenization, and stopword removal.
- **Scikit-learn:** For text vectorization (TF-IDF), data splitting, model training, and evaluation.
- **Seaborn & Matplotlib:** For data visualization, such as the Confusion Matrix.
- **Jupyter Notebook (via VS Code):** As the interactive development environment.

## 📈 Applied Methodology
The project followed a structured Data Science pipeline:

1.  **Text Cleaning and Preprocessing:**
    * Removal of HTML tags.
    * Removal of special characters and numbers.
    * Conversion to lowercase.
    * Tokenization, and stopword removal.

2.  **Feature Engineering:**
    * The cleaned text was converted into numerical vectors using the **TF-IDF** technique.
    * Two approaches were tested: using only single words (unigrams) and using both single words and word pairs (unigrams and bigrams).

3.  **Modeling and Comparative Evaluation:**
    * Three classification algorithms were trained and evaluated: **Logistic Regression**, **LinearSVC (SVM)**, and **Multinomial Naive Bayes**.
    * Performance was measured using Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.
    * To ensure reproducibility, the `random_state` parameter was fixed in both the data split and the models.

## 📊 Results
After experimentation, the performance of the models with the best feature configuration (TF-IDF with unigrams and bigrams) was as follows:

| Model |Accuracy | F1-Score (Avg) |
| **Logistic Regression** | **89.02%** | **0.89** |
| LinearSVC | 88.46% | 0.88 |
| Multinomial Naive Bayes | 84.98% | 0.85 |

### Further Analysis
An additional experiment with **lemmatization** was conducted. Interestingly, this step resulted in a slight performance drop (accuracy of 88.78% for the best model), suggesting that for this specific dataset and model, over-simplifying the text removed useful nuances for classification.

## ✅ Conclusion
The **Logistic Regression model, using TF-IDF vectorization with both unigrams and bigrams**, was selected as the final solution due to its superior performance, achieving a final accuracy of **89.02%**. The project demonstrates a robust NLP pipeline, from cleaning raw data to the comparative analysis and selection of the most effective model.

## 👨‍💻 Author

**[Rafael Dumar Batista]**

* **LinkedIn:** [https://www.linkedin.com/in/rafaeldumar/]
* **Email:** [rafaeldumar15@gmail.com]