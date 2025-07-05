import pandas as pd

# Load the sentiment data
sentiment_df = pd.read_csv('samsung_sentiment.csv')
print(f"Loaded {len(sentiment_df)} sentiment records")

# Load the stock data
stock_df = pd.read_csv('samsung_stock.csv')
print(f"Loaded {len(stock_df)} stock records")

# Convert date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_df['date'] = pd.to_datetime(stock_df['date'])

# Convert Close to numeric, filter out non-numeric rows, then assign
stock_df['Close_numeric'] = pd.to_numeric(stock_df['Close'], errors='coerce')
stock_df = stock_df[stock_df['Close_numeric'].notna()]
stock_df['Close'] = stock_df['Close_numeric']
stock_df = stock_df.drop(columns=['Close_numeric'])

# Group sentiment by date (average sentiment for each day)
daily_sentiment = sentiment_df.groupby('date')['sentiment'].mean().reset_index()
print(f"Daily sentiment data: {len(daily_sentiment)} days")

# Merge sentiment and stock data by date
merged_df = pd.merge(daily_sentiment, stock_df, on='date', how='inner')
print(f"Merged data: {len(merged_df)} days")

if len(merged_df) == 0:
    print("❌ No matching dates found between sentiment and stock data!")
    print("Sentiment date range:", sentiment_df['date'].min(), "to", sentiment_df['date'].max())
    print("Stock date range:", stock_df['date'].min(), "to", stock_df['date'].max())
else:
    # Calculate price change (next day's price - current price)
    merged_df['price_change'] = merged_df['Close'].shift(-1) - merged_df['Close']
    
    # Create target: 1 if price went up, 0 if down
    merged_df['target'] = (merged_df['price_change'] > 0).astype(int)
    
    # Remove the last row (no next day data)
    merged_df = merged_df.dropna()
    
    print(f"Final dataset: {len(merged_df)} days with targets")
    print("Sample of merged data:")
    print(merged_df[['date', 'sentiment', 'Close', 'price_change', 'target']].head())
    
    # Save to CSV
    merged_df.to_csv('merged_data.csv', index=False)
    print("✅ Saved merged_data.csv")