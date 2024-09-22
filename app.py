from flask import Flask, render_template, request, jsonify
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Preprocess text data to extract keywords
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"]]
    return keywords

# Use TF-IDF to rank and extract top keywords
def get_top_keywords(texts, n=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = X.toarray().sum(axis=0).argsort()[::-1]
    
    top_keywords = [feature_array[i] for i in tfidf_sorting[:n]]
    return top_keywords

# Sentiment Analysis function using VADER
def analyze_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score

# Route to handle text summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        
        # Extract keywords
        keywords = extract_keywords(feedback_text)
        
        # Get top 5 themes using TF-IDF
        top_keywords = get_top_keywords([feedback_text])
        
        # Analyze sentiment
        sentiment = analyze_sentiment(feedback_text)
        
        response = {
            "keywords": keywords,
            "top_keywords": top_keywords,
            "sentiment": sentiment
        }
        
        return jsonify(response)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
