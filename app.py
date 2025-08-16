from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def get_processed_stock_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        
        nltk.download('vader_lexicon')

    tweets_csv_file_path = os.path.expanduser('~/Desktop/archive/stock_tweets.csv')
    stock_csv_file_path = os.path.expanduser('~/Desktop/archive/stock_yfinance_data.csv')

    try:

        df_tweets = pd.read_csv(tweets_csv_file_path)


        sid = SentimentIntensityAnalyzer()


        def get_sentiment_scores(text):
            if pd.isna(text):
                return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
            return sid.polarity_scores(str(text))


        sentiment_scores = df_tweets['Tweet'].apply(get_sentiment_scores)
        df_tweets = pd.concat([df_tweets, sentiment_scores.apply(pd.Series)], axis=1)


        df_tweets['Date'] = pd.to_datetime(df_tweets['Date']).dt.date


        df_stock = pd.read_csv(stock_csv_file_path)


        df_stock['Date'] = pd.to_datetime(df_stock['Date']).dt.date




        daily_sentiment = df_tweets.groupby(['Date', 'Stock Name'])[['neg', 'neu', 'pos', 'compound']].mean().reset_index()

        # merge sentiment with stock data
        df_combined = pd.merge(daily_sentiment, df_stock, on=['Date', 'Stock Name'], how='outer')

        # sort
        df_combined = df_combined.sort_values(by=['Stock Name', 'Date']).reset_index(drop=True)

        
        sentiment_cols = ['neg', 'neu', 'pos', 'compound']
        for col in sentiment_cols:
            df_combined[col] = df_combined[col].fillna(0.0)


        stock_price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in stock_price_cols:
            df_combined[col] = df_combined.groupby('Stock Name')[col].ffill()


        df_combined['Next_Close'] = df_combined.groupby('Stock Name')['Close'].shift(-1)
        df_combined['Price_Direction'] = (df_combined['Next_Close'] > df_combined['Close']).astype(int)
        df_final = df_combined.dropna(subset=['Next_Close']).copy()

        # lagged sentiment/closed price
        df_final['compound_lag1'] = df_final.groupby('Stock Name')['compound'].shift(1)
        df_final['Close_lag1'] = df_final.groupby('Stock Name')['Close'].shift(1)
        df_final['Close_lag2'] = df_final.groupby('Stock Name')['Close'].shift(2)

        # add SMA
        df_final['SMA_5'] = df_final.groupby('Stock Name')['Close'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df_final['SMA_10'] = df_final.groupby('Stock Name')['Close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

        df_final.dropna(inplace=True)

        return df_final

    except FileNotFoundError as e:
        print(f"file not found")
        return pd.DataFrame() 
    except Exception as e:
        print(f"error")
        return pd.DataFrame() 

# process data
df_processed_data = get_processed_stock_data()


app = Flask(__name__)

@app.route('/')
def index():
    #stock names
    stock_names = df_processed_data['Stock Name'].unique().tolist()
    return render_template('index.html', stock_names=stock_names)

@app.route('/get_stock_data')
def get_stock_data():
    stock_name = request.args.get('stock')
    if stock_name:
        
        filtered_data = df_processed_data[df_processed_data['Stock Name'] == stock_name].copy()

        columns_to_display = ['Date', 'compound', 'Close', 'Price_Direction', 'SMA_5', 'SMA_10']
        filtered_data['Date'] = filtered_data['Date'].astype(str)
        return jsonify(filtered_data[columns_to_display].to_dict(orient='records'))
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)
