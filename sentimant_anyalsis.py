import pandas as pd
df = pd.read_csv("samsung_news.csv")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
def label_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df['label'] = df['sentiment'].apply(label_sentiment)
df['date'] = pd.to_datetime(df['date'])
df.to_csv("samsung_sentiment.csv", index=False)
print(f"✅ Analyzed sentiment for {len(df)} headlines")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")


