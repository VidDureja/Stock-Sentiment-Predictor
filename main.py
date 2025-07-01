#!/usr/bin/env python3
"""
Stock Sentiment Predictor - Main Pipeline
This script runs the complete workflow from data collection to model evaluation.
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("🚀 Stock Sentiment Predictor: End-to-End Pipeline")
    print("=" * 50)
    
    # Load the dataset
    news_df = pd.read_csv('Combined_News_DJIA.csv')

    # Combine all headlines for each day into a single string
    headline_columns = [col for col in news_df.columns if col.startswith('Top')]
    news_df['Combined_News'] = news_df[headline_columns].astype(str).agg(' '.join, axis=1)

    # Sentiment analysis
    print("Analyzing sentiment...")
    news_df['Sentiment'] = news_df['Combined_News'].apply(lambda x: float(TextBlob(x).sentiment.polarity))

    # Add previous day's DJIA movement as a feature
    news_df['Prev_Label'] = news_df['Label'].shift(1)
    news_df = news_df.dropna(subset=['Prev_Label'])

    # Prepare features and target
    X = news_df[['Sentiment', 'Prev_Label']]
    y = news_df['Label']  # 1 = DJIA up, 0 = DJIA down

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    print("Training model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred)) 
    
    print("\n🎉 Done! You can now see your results above.")
    print("\n📊 Next steps you can take:")
    print("1. Improve the model with more features")
    print("2. Add visualization of results")
    print("3. Implement real-time predictions")
    print("4. Add more sophisticated ML algorithms")

if __name__ == "__main__":
    main()