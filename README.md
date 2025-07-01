# Stock Market Sentiment Predictor

Predicting the Dow Jones movement using news headlines and sentiment analysis.

## Project Overview
This project explores whether daily news headlines can help predict if the stock market (DJIA) will go up or down. It covers the full data science process: loading data, analyzing sentiment, building features, training a model, and checking results.

## Data
- Source: [Kaggle - Daily News for Stock Market Prediction](https://www.kaggle.com/datasets/aaron7sun/stocknews)
- Contains: Daily news headlines and DJIA up/down labels (2008–2016)

## Approach
- Combine all news headlines for each day into one string
- Use TextBlob to score the sentiment (positive/negative) of the news
- Add the previous day's DJIA movement as a feature
- Train a Random Forest model to predict up/down
- Evaluate with accuracy and a classification report

## How to Run
1. Clone this repo and open the folder
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the Kaggle dataset and put `Combined_News_DJIA.csv` in the folder
4. Run:
   ```sh
   python main.py
   ```

## Results
- Accuracy: ~52% (a bit better than random guessing)
- Shows the challenge of predicting markets with news sentiment alone

## What's Next
- Try more features (like rolling averages, headline counts, or advanced NLP)
- Experiment with other models
- Add plots or dashboards to visualize results

---

*Made by [Your Name]*
