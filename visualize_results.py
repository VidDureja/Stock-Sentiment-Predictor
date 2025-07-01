import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the merged data"""
    try:
        df = pd.read_csv("merged_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("❌ merged_data.csv not found. Please run the pipeline first.")
        return None

def plot_sentiment_over_time(df):
    """Plot sentiment scores over time"""
    plt.figure(figsize=(15, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Sentiment over time
    ax1.plot(df['date'], df['sentiment'], alpha=0.7, linewidth=1)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('📈 Sentiment Score Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sentiment Score')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stock price over time
    ax2.plot(df['date'], df['Close'], color='green', alpha=0.7, linewidth=1)
    ax2.set_title('💰 Stock Price Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Stock Price ($)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentiment_and_price_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sentiment_distribution(df):
    """Plot distribution of sentiment scores"""
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Histogram of sentiment scores
    ax1.hist(df['sentiment'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_title('📊 Distribution of Sentiment Scores', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    ax2.boxplot(df['sentiment'])
    ax2.set_title('📦 Sentiment Score Box Plot', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Sentiment Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_analysis(df):
    """Plot correlation between sentiment and price changes"""
    # Calculate daily price changes
    df['price_change'] = df['Close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Scatter plot of sentiment vs price change
    ax1.scatter(df['sentiment'], df['price_change'], alpha=0.6)
    ax1.set_title('🔄 Sentiment vs Price Change', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Price Change (%)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of sentiment vs absolute price change
    ax2.scatter(df['sentiment'], df['price_change_abs'], alpha=0.6, color='orange')
    ax2.set_title('📈 Sentiment vs Absolute Price Change', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment Score')
    ax2.set_ylabel('Absolute Price Change (%)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling correlation
    window = 30
    rolling_corr = df['sentiment'].rolling(window).corr(df['price_change'])
    ax3.plot(df['date'], rolling_corr, linewidth=2)
    ax3.set_title(f'🔄 {window}-Day Rolling Correlation', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sentiment by price movement direction
    df['price_direction'] = np.where(df['price_change'] > 0, 'Up', 'Down')
    sentiment_by_direction = df.groupby('price_direction')['sentiment'].mean()
    ax4.bar(sentiment_by_direction.index, sentiment_by_direction.values, 
            color=['green', 'red'], alpha=0.7)
    ax4.set_title('📊 Average Sentiment by Price Direction', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Average Sentiment Score')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_stats(df):
    """Print summary statistics"""
    print("\n📊 SUMMARY STATISTICS")
    print("=" * 50)
    
    print(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"📈 Total Days: {len(df)}")
    
    print(f"\n💰 Stock Price:")
    print(f"   Average: ${df['Close'].mean():.2f}")
    print(f"   Min: ${df['Close'].min():.2f}")
    print(f"   Max: ${df['Close'].max():.2f}")
    
    print(f"\n😊 Sentiment Analysis:")
    print(f"   Average: {df['sentiment'].mean():.3f}")
    print(f"   Min: {df['sentiment'].min():.3f}")
    print(f"   Max: {df['sentiment'].max():.3f}")
    print(f"   Std Dev: {df['sentiment'].std():.3f}")
    
    # Sentiment distribution
    positive = (df['sentiment'] > 0.05).sum()
    negative = (df['sentiment'] < -0.05).sum()
    neutral = len(df) - positive - negative
    
    print(f"\n📊 Sentiment Distribution:")
    print(f"   Positive: {positive} days ({positive/len(df)*100:.1f}%)")
    print(f"   Negative: {negative} days ({negative/len(df)*100:.1f}%)")
    print(f"   Neutral: {neutral} days ({neutral/len(df)*100:.1f}%)")
    
    # Correlation
    correlation = df['sentiment'].corr(df['Close'].pct_change())
    print(f"\n🔄 Correlation (Sentiment vs Price Change): {correlation:.3f}")

def main():
    print("📊 Stock Sentiment Predictor - Visualization")
    print("=" * 50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Print summary statistics
    print_summary_stats(df)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    plot_sentiment_over_time(df)
    plot_sentiment_distribution(df)
    plot_correlation_analysis(df)
    
    print("\n✅ Visualizations completed!")
    print("📁 Saved plots as PNG files in current directory")

if __name__ == "__main__":
    main() 