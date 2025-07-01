import pandas as pd

# Print out columns to debug
df1 = pd.read_csv("samsung_sentiment.csv")
df2 = pd.read_csv("samsung_stock.csv")

print("📄 Sentiment CSV Columns:", df1.columns)
print("📄 Stock CSV Columns:", df2.columns)

# Load sentiment data
sentiment_df = pd.read_csv("tech_sentiment.csv")

# Load stock price data
stock_df = pd.read_csv("microsoft_stock.csv")

# Aggregate sentiment by month (mean sentiment across all companies)
monthly_sentiment = sentiment_df.groupby('month')['sentiment'].mean().reset_index()

# Aggregate stock by month (last close price)
monthly_stock = stock_df.sort_values('date').groupby('month').agg({'date':'last', 'Close':'last'}).reset_index()
monthly_stock['Close'] = pd.to_numeric(monthly_stock['Close'], errors='coerce')

# Compute monthly price change as target
monthly_stock['price_change'] = monthly_stock['Close'].diff().shift(-1)
monthly_stock['target'] = monthly_stock['price_change'].apply(lambda x: 1 if x > 0 else 0)

# Merge on month
merged_df = pd.merge(monthly_sentiment, monthly_stock, on='month', how='inner')

print(merged_df.head())
merged_df.to_csv("merged_data.csv", index=False)