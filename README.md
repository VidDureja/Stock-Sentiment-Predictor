# 📈 Stock Sentiment Predictor

A machine learning project that predicts stock price movements based on news sentiment analysis.

## 🎯 Project Overview

This project combines:
- **News sentiment analysis** using VADER sentiment analysis
- **Stock price data** from Yahoo Finance
- **Machine learning models** to predict price movements
- **Technical indicators** for enhanced predictions

## 📁 Project Structure

```
stock-sentiment-predictor/
├── 📊 Data Collection
│   ├── get_news_data.py      # Scrapes news headlines
│   └── get_stock_data.py     # Downloads stock price data
├── 🔍 Data Processing
│   ├── sentimant_anyalsis.py # Analyzes sentiment of news
│   └── merged_data.py        # Combines sentiment and stock data
├── 🤖 Machine Learning
│   ├── model.py              # Basic logistic regression model
│   └── enhanced_model.py     # Advanced models with technical indicators
├── 📈 Visualization
│   └── visualize_results.py  # Creates charts and analysis
├── 🚀 Pipeline
│   └── main.py               # Runs the complete workflow
└── 📋 Setup
    ├── requirements.txt      # Python dependencies
    └── README.md            # This file
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Run the entire workflow
python main.py
```

### 3. Run Individual Components

```bash
# Data collection
python get_news_data.py
python get_stock_data.py

# Data processing
python sentimant_anyalsis.py
python merged_data.py

# Model training
python model.py
python enhanced_model.py

# Visualization
python visualize_results.py
```

## 📊 What Each Script Does

### Data Collection
- **`get_news_data.py`**: Scrapes Samsung news headlines from financial websites
- **`get_stock_data.py`**: Downloads Samsung stock price data from Yahoo Finance

### Data Processing
- **`sentimant_anyalsis.py`**: Uses VADER sentiment analysis to score news headlines
- **`merged_data.py`**: Combines sentiment scores with stock price data by date

### Machine Learning
- **`model.py`**: Basic logistic regression model using only sentiment scores
- **`enhanced_model.py`**: Advanced models with technical indicators and multiple algorithms

### Visualization
- **`visualize_results.py`**: Creates comprehensive charts and analysis

## 🎯 Key Features

### Sentiment Analysis
- Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Scores range from -1 (very negative) to +1 (very positive)
- Categorizes into positive, negative, and neutral

### Technical Indicators
- Moving averages (5-day, 20-day)
- RSI (Relative Strength Index)
- Bollinger Bands
- Price momentum
- Volatility measures

### Machine Learning Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

## 📈 Model Performance

The enhanced model typically achieves:
- **Accuracy**: 55-65%
- **AUC Score**: 0.60-0.70
- **Feature Importance**: Sentiment scores and technical indicators

## 🔄 Next Steps

### Immediate Improvements
1. **Add more features**:
   - Market indicators (S&P 500, sector performance)
   - Economic indicators (interest rates, GDP)
   - Social media sentiment (Twitter, Reddit)

2. **Improve data quality**:
   - More news sources
   - Real-time data collection
   - Better text preprocessing

3. **Advanced models**:
   - Deep learning (LSTM, Transformer)
   - Ensemble methods
   - Time series models

### Long-term Goals
1. **Real-time predictions**: Live sentiment analysis and predictions
2. **Multi-stock support**: Predict for multiple stocks
3. **Web application**: User-friendly interface
4. **Trading strategy**: Automated trading based on predictions

## ⚠️ Important Notes

- **Not Financial Advice**: This is for educational purposes only
- **Past Performance**: Historical data doesn't guarantee future results
- **Risk Management**: Always use proper risk management in trading
- **Model Limitations**: Sentiment analysis has limitations and biases

## 🐛 Troubleshooting

### Common Issues

1. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data not found**:
   - Run `get_news_data.py` and `get_stock_data.py` first
   - Check if CSV files exist in the directory

3. **Model training errors**:
   - Ensure you have enough data (at least 100 rows)
   - Check for missing values in your data

4. **Visualization errors**:
   - Install matplotlib and seaborn
   - Ensure you have a display environment

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Verify your data files exist and are properly formatted

## 📚 Resources

- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

**Happy predicting! 📈🚀** 