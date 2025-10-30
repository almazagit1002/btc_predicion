# Machine Learning Models for BTC Price Prediction

## Problem Formulation Options

### Option 1: Classification (Recommended to Start)
**Predict Direction**: Will BTC go up or down tomorrow/next week?
- **Target**: Binary (1 = price increase, 0 = price decrease)
- **Easier to evaluate** and more practical for trading decisions
- **Success metric**: Accuracy > 55% is profitable (accounting for fees)

### Option 2: Regression
**Predict Magnitude**: How much will BTC price change?
- **Target**: Continuous (e.g., +3.2%, -1.8%)
- **More complex** but potentially more profitable
- **Success metric**: Mean Absolute Error, R-squared

### Option 3: Multi-class Classification
**Predict Ranges**: Strong up, weak up, flat, weak down, strong down
- **Target**: 3-5 categories
- **Good middle ground** between classification and regression

## Feature Engineering Strategy

### Core Sector Features
```python
# Daily returns
xlk_return_1d = (xlk_close / xlk_close.shift(1)) - 1
xle_return_1d = (xle_close / xle_close.shift(1)) - 1
# ... for all sectors

# Multi-day returns (momentum)
xlk_return_3d = (xlk_close / xlk_close.shift(3)) - 1
xlk_return_7d = (xlk_close / xlk_close.shift(7)) - 1

# Relative performance
xlk_vs_spy = xlk_return_1d - spy_return_1d
```

### Technical Indicators
```python
# Moving averages
xlk_ma_20 = xlk_close.rolling(20).mean()
xlk_above_ma = (xlk_close > xlk_ma_20).astype(int)

# RSI for each sector
xlk_rsi = calculate_rsi(xlk_close, 14)

# Sector momentum
xlk_momentum = xlk_close / xlk_close.shift(20) - 1
```

### Cross-Asset Features
```python
# Correlations (rolling)
xlk_btc_corr_30d = btc_return.rolling(30).corr(xlk_return)

# Relative volatility
xlk_vol_ratio = xlk_volatility / spy_volatility

# Beta coefficients (rolling)
xlk_beta = rolling_beta(xlk_return, spy_return, window=60)
```

### Seasonality Features
```python
# Calendar features
month = data.index.month
quarter = data.index.quarter
day_of_week = data.index.dayofweek
day_of_month = data.index.day

# Cyclical encoding (preserves relationships)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# Market-specific seasons
is_earnings_season = ((month.isin([1, 4, 7, 10])) & (day_of_month <= 15)).astype(int)
is_year_end = (month == 12).astype(int)
```

### Macro Environment Features
```python
# Market regime indicators
vix_level = vix_close
vix_spike = (vix_close > vix_close.rolling(20).mean() + 2*vix_close.rolling(20).std()).astype(int)

# Interest rate environment
yield_curve_slope = ten_year_yield - two_year_yield
fed_meeting_week = get_fed_meeting_dates()  # Binary indicator
```

## Model Options (Ranked by Complexity)

### Tier 1: Simple & Interpretable

#### 1. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

# Pros:
- Highly interpretable
- Fast training/prediction
- Good baseline model
- Shows feature importance clearly

# Cons:
- Assumes linear relationships
- May miss complex patterns

# Use when:
- You want to understand which sectors matter most
- Starting your analysis
- Need explainable results
```

#### 2. Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Pros:
- Handles non-linear relationships
- Built-in feature importance
- Robust to outliers
- Good performance out-of-the-box

# Cons:
- Can overfit with small datasets
- Less interpretable than logistic regression

# Use when:
- You have enough data (>2 years daily)
- Want good performance with minimal tuning
```

### Tier 2: Advanced Traditional ML

#### 3. XGBoost/LightGBM
```python
import xgboost as xgb
import lightgbm as lgb

# Pros:
- Often best performance on tabular data
- Excellent feature importance
- Handles missing values well
- Good for time series

# Cons:
- Requires hyperparameter tuning
- Can overfit easily
- More complex to interpret

# Use when:
- You want maximum predictive power
- Have time for hyperparameter optimization
- Competition/production model
```

#### 4. Support Vector Machine (SVM)
```python
from sklearn.svm import SVC

# Pros:
- Good with high-dimensional data
- Works well with limited data
- Robust to outliers

# Cons:
- Slow on large datasets
- Requires feature scaling
- Hard to interpret

# Use when:
- You have many features relative to samples
- Linear models perform poorly
```

### Tier 3: Deep Learning

#### 5. Neural Networks (MLP)
```python
import tensorflow as tf

# Pros:
- Can learn complex patterns
- Flexible architecture
- Good for non-linear relationships

# Cons:
- Needs lots of data
- Black box (hard to interpret)
- Prone to overfitting

# Use when:
- You have >5 years of daily data
- Traditional ML plateaus
- Can afford black box approach
```

#### 6. LSTM/GRU (Sequence Models)
```python
from tensorflow.keras.layers import LSTM

# Pros:
- Designed for time series
- Can capture long-term dependencies
- State-of-the-art for sequences

# Cons:
- Very complex
- Needs massive amounts of data
- Difficult to tune

# Use when:
- You have >10 years of data
- Sequence patterns are crucial
- Have ML engineering resources
```

## Recommended Implementation Path

### Phase 1: Foundation (Week 1-2)
```python
# Start with Logistic Regression
target = (btc_close.shift(-1) > btc_close).astype(int)  # Next day up/down

features = [
    'xlk_return_1d', 'xle_return_1d', 'xlp_return_1d',  # Sector returns
    'xlk_return_7d', 'xle_return_7d',  # Momentum
    'month_sin', 'month_cos', 'day_of_week',  # Seasonality
    'vix_level', 'vix_spike'  # Market regime
]

# Simple train/validation split
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train and evaluate
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

### Phase 2: Enhancement (Week 3-4)
```python
# Add Random Forest for comparison
# Add more engineered features
# Implement proper time series cross-validation
# Feature selection and importance analysis
```

### Phase 3: Advanced (Month 2)
```python
# XGBoost with hyperparameter tuning
# Ensemble methods (combine multiple models)
# Advanced feature engineering
# Multiple prediction horizons (1-day, 3-day, 7-day)
```

## Key Success Factors

### 1. Proper Cross-Validation
```python
# Time Series Split (CRITICAL!)
from sklearn.model_selection import TimeSeriesSplit

# Don't use random splits - use chronological
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

### 2. Feature Importance Analysis
```python
# Understand what drives predictions
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Top predictors might be:
# xlk_return_1d: 0.25  (tech sector most important)
# month_sin: 0.15      (seasonality matters)
# vix_spike: 0.12      (volatility regime)
```

### 3. Performance Metrics
```python
# For classification
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Trading-specific metrics
win_rate = accuracy
profit_factor = sum(profits) / sum(losses)
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
```

## Expected Performance Benchmarks

### Realistic Expectations
- **Random Baseline**: 50% accuracy
- **Simple Model**: 52-55% accuracy
- **Good Model**: 55-58% accuracy
- **Excellent Model**: 58-62% accuracy
- **Suspicious (likely overfit)**: >65% accuracy

### Model-Specific Expectations
| Model Type | Expected Accuracy | Training Time | Interpretability |
|------------|------------------|---------------|------------------|
| Logistic Regression | 52-55% | Minutes | High |
| Random Forest | 54-57% | Minutes | Medium |
| XGBoost | 55-59% | Hours | Medium |
| Neural Network | 55-60% | Hours-Days | Low |
| LSTM | 56-61% | Days | Very Low |

## Practical Next Steps

### Start Simple
1. **Logistic Regression** with sector returns + basic seasonality
2. **Evaluate performance** using proper time series validation
3. **Analyze feature importance** to understand what works

### Scale Up
1. **Add Random Forest** to capture non-linearities
2. **Engineer more features** based on initial insights
3. **Try XGBoost** for maximum performance

### Production Considerations
- **Model retraining** schedule (monthly/quarterly)
- **Feature drift** monitoring
- **Performance tracking** over time
- **Risk management** integration

Remember: A 55% accuracy model that consistently beats random chance is extremely valuable in financial markets!