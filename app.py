from flask import Flask, render_template, request, jsonify
from langdetect import detect
import spacy
from collections import Counter
from transformers import pipeline, MBartForConditionalGeneration, MBart50Tokenizer, BartForConditionalGeneration, BartTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load spaCy models for English and Japanese
nlp_en = spacy.load("en_core_web_sm")  # English
nlp_ja = spacy.load("ja_core_news_sm")  # Japanese

# Initialize sentiment analysis pipelines with optimized models
sentiment_pipeline_en = pipeline("sentiment-analysis", model="distilbert-base-uncased", use_fast=True)  # English
sentiment_pipeline_ja = pipeline("sentiment-analysis", model="cl-tohoku/bert-base-japanese", use_fast=True)  # Japanese

# Initialize mBART for summarization (multilingual, optimized with fast tokenizer)
mbart_tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_fast=True)
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Initialize BART for summarization (English only)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Language detection function
def detect_language(text):
    return detect(text)  # Returns 'en' for English, 'ja' for Japanese

# Tokenize text and return top 10 keywords with their frequencies
def extract_top_keywords(text, lang_code, n=10):
    if lang_code == 'en':
        doc = nlp_en(text)
    elif lang_code == 'ja':
        doc = nlp_ja(text)
    else:
        return []  # Fallback for unsupported languages

    # Extract nouns, adjectives, and verbs
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"]]
    
    # Count keyword frequencies
    keyword_freq = Counter(keywords)
    
    # Get the top n keywords with their frequencies
    top_keywords = keyword_freq.most_common(n)
    
    return top_keywords

# Sentiment analysis function with language-specific models
def analyze_sentiment(text, lang_code):
    if lang_code == 'en':
        result = sentiment_pipeline_en(text)
    elif lang_code == 'ja':
        result = sentiment_pipeline_ja(text)
    else:
        return {"label": "Unsupported language", "score": 0.0}
    
    # Mapping the labels for easier interpretation
    label_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    
    sentiment_label = label_mapping.get(result[0]['label'], "Unknown")
    sentiment_score = result[0]['score']
    
    return {"label": sentiment_label, "score": sentiment_score}

# Summarization function using BART (English only)
def summarize_text_with_bart(text):
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)  # Limit input length
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Summarization function using mBART (Multilingual)
def summarize_text_with_mbart(text, lang_code):
    # Use correct mBART50 language codes
    if lang_code == 'en':
        lang_code = 'en_XX'
    elif lang_code == 'ja':
        lang_code = 'ja_XX'
    else:
        return "Unsupported language"

    mbart_tokenizer.src_lang = lang_code
    inputs = mbart_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)  # Limit input length

    summary_ids = mbart_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = mbart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Route to handle text submission and keyword extraction, sentiment analysis, and summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        feedback_text = request.form['feedback']

        # Detect language of the text
        lang_code = detect_language(feedback_text)

        # Extract top keywords
        top_keywords = extract_top_keywords(feedback_text, lang_code)

        # Analyze sentiment
        sentiment = analyze_sentiment(feedback_text, lang_code)

        # Summarize the text
        if lang_code == 'en':  # Use BART for English
            summary = summarize_text_with_bart(feedback_text)
        else:  # Use mBART for other languages like Japanese
            summary = summarize_text_with_mbart(feedback_text, lang_code)

        # Prepare response
        response = {
            "language": lang_code,
            "top_keywords": top_keywords,
            "sentiment": sentiment,
            "summary": summary
        }

        return jsonify(response)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
