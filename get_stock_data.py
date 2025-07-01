import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define stock and time period
ticker = "MSFT"
start_date = "2024-06-01"
end_date = "2025-05-31"

# Download stock price data
data = yf.download(ticker, start=start_date, end=end_date)

if data is None or data.empty:
    print("Failed to download stock data. Please check the ticker or date range.")
    exit()

# Show first 5 rows
print(data.head())

# Plot closing price
data["Close"].plot(title=f"{ticker} Closing Price", figsize=(10, 5))
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

data.reset_index(inplace=True)
data.rename(columns={"Date": "date"}, inplace=True)
data['month'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m')
data.to_csv("microsoft_stock.csv")




