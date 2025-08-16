import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')


tweets_csv_file_path = os.path.expanduser('~/Desktop/archive/stock_tweets.csv')
stock_csv_file_path = os.path.expanduser('~/Desktop/archive/stock_yfinance_data.csv')

try:
    # tweets to pandas
    df_tweets = pd.read_csv(tweets_csv_file_path)

    # VADER 
    sid = SentimentIntensityAnalyzer()

    # sentiment scores
    def get_sentiment_scores(text):
        if pd.isna(text):
            return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
        return sid.polarity_scores(str(text))

    # sentiment analysis to tweets
    sentiment_scores = df_tweets['Tweet'].apply(get_sentiment_scores)
    df_tweets = pd.concat([df_tweets, sentiment_scores.apply(pd.Series)], axis=1)


    df_tweets['Date'] = pd.to_datetime(df_tweets['Date']).dt.date

    # stock data to pandas
    df_stock = pd.read_csv(stock_csv_file_path)


    df_stock['Date'] = pd.to_datetime(df_stock['Date']).dt.date

    
    daily_sentiment = df_tweets.groupby(['Date', 'Stock Name'])[['neg', 'neu', 'pos', 'compound']].mean().reset_index()


    df_combined = pd.merge(daily_sentiment, df_stock, on=['Date', 'Stock Name'], how='outer')


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

   
    df_final['compound_lag1'] = df_final.groupby('Stock Name')['compound'].shift(1)
    df_final['Close_lag1'] = df_final.groupby('Stock Name')['Close'].shift(1)
    df_final['Close_lag2'] = df_final.groupby('Stock Name')['Close'].shift(2)

    # add SMAs
    
    df_final['SMA_5'] = df_final.groupby('Stock Name')['Close'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df_final['SMA_10'] = df_final.groupby('Stock Name')['Close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())


    df_final.dropna(inplace=True)

#model training

    
    # SMA_5 / SMA_10 
    features = ['compound', 'compound_lag1', 'Close_lag1', 'Close_lag2', 'SMA_5', 'SMA_10']
    target = 'Price_Direction'

    X = df_final[features]
    y = df_final[target]

    df_final_sorted = df_final.sort_values(by='Date')
    
    # split point
    split_index = int(len(df_final_sorted) * 0.8)
    X_train = df_final_sorted.iloc[:split_index][features]
    X_test = df_final_sorted.iloc[split_index:][features]
    y_train = df_final_sorted.iloc[:split_index][target]
    y_test = df_final_sorted.iloc[split_index:][target]

    #feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
   

    print(f"Training set size: {len(X_train_scaled)} samples")
    print(f"Test set size: {len(X_test_scaled)} samples")


    
    model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    model.fit(X_train_scaled, y_train)


    # test set predictions

    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

except FileNotFoundError as e:
    print(f"file not found: {e}")
except Exception as e:
    print(f"error{e}")
