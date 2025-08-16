# Stock Sentiment Analysis
This project analyzes the sentiment of stock-related tweets and correlates it with historical stock prices, aiming to predict market direction using machine learning. It includes a Flask web application for interactive exploration and a machine learning pipeline for price direction prediction.

# Features
- Sentiment Analysis: Uses NLTK’s VADER to analyze tweet sentiment for each stock and date.
- Data Processing: Merges daily average sentiment scores with Yahoo Finance stock data.
- Feature Engineering: Computes rolling averages (SMA), lags, and direction labels for supervised learning.
- Machine Learning: Trains a Random Forest classifier to predict if the next day's close price will go up or down.
- Web App: Interactive dashboard to select stocks and view sentiment/price features.


# Installation
1. Clone the repository 
    git clone https://github.com/at625/stock_sentiment_analysis.git
    cd stock_sentiment_analysis
2. Install dependencies
   pip install pandas scikit-learn nltk flask
     - Download the NLTK VADER lexicon if not present (handled automatically in code)

3. Prepare data
   Place your CSV files at:
   ~/Desktop/archive/stock_tweets.csv
   ~/Desktop/archive/stock_yfinance_data.csv

# Usage
Web Dashboard
1. Run the Flask app:
    - python app.py
2. Open your browser at http://127.0.0.1:5000
3. Select a stock to view sentiment and price metrics.

# Main Files
- app.py: Loads, processes, and serves stock sentiment data via Flask web routes.
- stock_sentiment_analyzer.py: Loads, processes data and trains/evaluates the Random Forest model.
- index.html: Frontend template for the dashboard, shows data in a table.

# Technologies

Python (Flask, pandas, scikit-learn, NLTK)
HTML/CSS/JavaScript (for the dashboard UI)
