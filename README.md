# Stock Market Sentiment Predictor

Predicting Samsung stock movement using news headlines and sentiment analysis.

## Project Overview

This project explores whether daily news headlines about Samsung can help predict if Samsung's stock price will go up or down.  
It covers the full data science process: loading data, analyzing sentiment, building features, training a model, and checking results.

## Data

- **News:** Sampled or real Samsung news headlines (from `samsung_news.csv`)
- **Stock:** Samsung daily closing prices (from `samsung_stock.csv`)
- **Sentiment:** Calculated using TextBlob (`samsung_sentiment.csv`)

## Approach

- Collect Samsung news headlines and stock data for matching dates
- Use TextBlob to score the sentiment (positive/negative) of the news
- Merge sentiment and stock data by date
- Train a Logistic Regression model to predict if the stock will go up or down the next day
- Evaluate with accuracy and a classification report

## How to Run

1. Clone this repo and open the folder
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the scripts in order:
    ```bash
    python get_news_data.py
    python sentimant_anyalsis.py
    python get_stock_data.py
    python merged_data.py
    python model.py
    ```
4. View the results in your terminal

## Results

- With the sample data, the model trains and predicts up/down movement
- Accuracy and sentiment coefficient are shown in the terminal

## What's Next

- Try more features (like technical indicators, more news sources, etc.)
- Experiment with other models (Random Forest, SVM, etc.)
- Add plots or dashboards to visualize results

---

_Made by Vidhit Dureja_
