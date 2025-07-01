import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # Price momentum
    df['price_momentum_5'] = df['Close'].pct_change(5)
    df['price_momentum_10'] = df['Close'].pct_change(10)
    
    # Volatility
    df['volatility_5'] = df['Close'].rolling(window=5).std()
    df['volatility_20'] = df['Close'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_upper'] = df['MA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['MA_20'] - (df['Close'].rolling(window=20).std() * 2)
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume indicators (if available)
    if 'Volume' in df.columns:
        df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_5']
    
    return df

def add_sentiment_features(df):
    """Add sentiment-based features"""
    # Sentiment momentum
    df['sentiment_momentum_3'] = df['sentiment'].rolling(window=3).mean()
    df['sentiment_momentum_7'] = df['sentiment'].rolling(window=7).mean()
    
    # Sentiment volatility
    df['sentiment_volatility_5'] = df['sentiment'].rolling(window=5).std()
    
    # Sentiment change
    df['sentiment_change'] = df['sentiment'].diff()
    df['sentiment_change_abs'] = df['sentiment_change'].abs()
    
    # Sentiment categories
    df['sentiment_positive'] = (df['sentiment'] > 0.05).astype(int)
    df['sentiment_negative'] = (df['sentiment'] < -0.05).astype(int)
    df['sentiment_neutral'] = ((df['sentiment'] >= -0.05) & (df['sentiment'] <= 0.05)).astype(int)
    
    return df

def prepare_features(df):
    """Prepare all features for modeling"""
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Add sentiment features
    df = add_sentiment_features(df)
    
    # Create target variable
    df['price_movement'] = df['Close'].diff().shift(-1)
    df['target'] = df['price_movement'].apply(lambda x: 1 if x > 0 else 0)
    
    # Select features
    feature_columns = [
        'sentiment', 'sentiment_momentum_3', 'sentiment_momentum_7',
        'sentiment_volatility_5', 'sentiment_change', 'sentiment_change_abs',
        'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
        'MA_5', 'MA_20', 'price_momentum_5', 'price_momentum_10',
        'volatility_5', 'volatility_20', 'RSI', 'BB_position'
    ]
    
    # Add volume features if available
    if 'Volume' in df.columns:
        feature_columns.extend(['volume_ratio'])
    
    # Remove rows with missing values
    df = df.dropna()
    
    return df, feature_columns

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC Score: {auc_score:.3f}")
    
    return results

def print_feature_importance(model, feature_names):
    """Print feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

def main():
    print("🚀 Enhanced Stock Sentiment Predictor")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_csv("merged_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded {len(df)} rows of data")
    except FileNotFoundError:
        print("❌ merged_data.csv not found. Please run the pipeline first.")
        return
    
    # Prepare features
    print("\n🔧 Preparing features...")
    df, feature_columns = prepare_features(df)
    print(f"✅ Prepared {len(feature_columns)} features")
    print(f"📊 Final dataset size: {len(df)} rows")
    
    # Split data
    X = df[feature_columns]
    y = df['target']
    
    print(f"\n📈 Target distribution:")
    print(y.value_counts())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🧪 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.3f}")
    print(f"   AUC Score: {results[best_model_name]['auc_score']:.3f}")
    
    # Feature importance
    print_feature_importance(best_model, feature_columns)
    
    # Detailed classification report for best model
    print(f"\n📊 Detailed Classification Report for {best_model_name}:")
    print(classification_report(y_test, results[best_model_name]['predictions']))
    
    # Save the best model
    import joblib
    joblib.dump(best_model, 'best_sentiment_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print(f"\n💾 Saved best model as 'best_sentiment_model.pkl'")
    print(f"💾 Saved feature scaler as 'feature_scaler.pkl'")
    
    print("\n✅ Enhanced model training completed!")

if __name__ == "__main__":
    main() 