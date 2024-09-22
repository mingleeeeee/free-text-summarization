# Free-Text Summarization and Sentiment Analysis Web App

This project is a Flask-based web application that processes free-text feedback and performs the following tasks:
- **Keyword Extraction**: Extracts the top 10 keywords along with their frequencies.
- **Sentiment Analysis**: Uses XLM-Roberta to analyze the sentiment of the feedback (Positive/Negative/Neutral).
- **Text Summarization**: Summarizes the input text using mBART (for multilingual support) or BART (for English).

## Features

1. **Language Detection**: Automatically detects if the input text is in English or Japanese.
2. **Keyword Extraction**: Extracts the most relevant keywords based on their frequency.
3. **Sentiment Analysis**: Provides sentiment classification for the input text.
4. **Summarization**: Summarizes the input text using advanced NLP models.
5. **Multilingual Support**: Uses mBART for multilingual summarization (currently supports English and Japanese).

## Demo

- Input your feedback in English or Japanese.
- Get the top keywords, sentiment analysis, and summarized version of your text.
  
## Installation

### Prerequisites

- Python 3.8+
- `pip` package manager

### Clone the repository

```bash
git clone https://github.com/mingleeeeee/free-text-summarization
cd free-text-summarization
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # English model
python -m spacy download ja_core_news_sm  # Japanese model
```
