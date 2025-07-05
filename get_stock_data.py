import yfinance as yf
import pandas as pd

# Define stock and time period
# Samsung Electronics Co Ltd (KRX: 005930.KS)
ticker = "005930.KS"
start_date = "2025-01-01"
end_date = "2025-12-31"

# Download stock price data
data = yf.download(ticker, start=start_date, end=end_date)

if data is None or data.empty:
    print("Failed to download stock data. Please check the ticker or date range.")
    exit()

# Show first 5 rows
print(data.head())

# Prepare for saving
# Reset index to get date as a column
data.reset_index(inplace=True)
data.rename(columns={"Date": "date"}, inplace=True)

# Save to CSV
cols = ["date", "Close", "High", "Low", "Open", "Volume"]
data = data[cols]
data.to_csv("samsung_stock.csv", index=False)
print(f"Saved {len(data)} rows to samsung_stock.csv")




