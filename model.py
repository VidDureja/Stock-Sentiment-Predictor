import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 📥 Load the merged dataset
try:
    df = pd.read_csv("merged_data.csv")
    print(f"✅ Loaded {len(df)} rows of merged data")
except FileNotFoundError:
    print("❌ merged_data.csv not found. Please run merged_data.py first.")
    exit()

# Check for missing values
print(f"Missing values in sentiment: {df['sentiment'].isnull().sum()}")
print(f"Missing values in Close: {df['Close'].isnull().sum()}")

# Remove rows with missing values
df = df.dropna()
print(f"After removing missing values: {len(df)} rows")

# 📊 Create target column: 1 if next day's price went up, else 0
df["price_movement"] = df["Close"].diff().shift(-1)
df["target"] = df["price_movement"].apply(lambda x: 1 if x > 0 else 0)

# Remove the last row since we can't predict it (no next day data)
df = df.dropna()
print(f"Final dataset size: {len(df)} rows")

# 🎯 Features and labels
X = df[["sentiment"]]      # Sentiment score is the only input
y = df["target"]           # Target is whether price went up next day

print(f"Feature shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")

if len(df) < 2:
    print("Not enough data to train/test a model. Please collect more data.")
    exit()

# 🧪 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"\n📊 Model Performance:")
print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")

# More detailed evaluation
y_pred = model.predict(X_test)
print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Show feature importance
print(f"\n🎯 Sentiment coefficient: {model.coef_[0][0]:.4f}")
print("✅ Script completed successfully.")
