import pandas as pd

df1 = pd.read_csv("samsung_sentiment.csv")
df2 = pd.read_csv("samsung_stock.csv")

print("📄 Sentiment CSV Columns:", df1.columns.tolist())
print("📄 Stock CSV Columns:", df2.columns.tolist())
