import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

def add_btc_technical_indicators(df: pd.DataFrame, price_col: str = 'cryptos_BTC') -> pd.DataFrame:
    df = df.copy()
    
    # Returns
    df['btc_return_1d'] = df[price_col].pct_change(1)
    df['btc_return_7d'] = df[price_col].pct_change(7)
    df['btc_return_30d'] = df[price_col].pct_change(30)
    
    # Rolling volatility (standard deviation of returns)
    df['btc_volatility_7d'] = df['btc_return_1d'].rolling(window=7).std()
    df['btc_volatility_30d'] = df['btc_return_1d'].rolling(window=30).std()
    
    # Moving averages
    df['btc_moving_avg_7d'] = df[price_col].rolling(window=7).mean()
    df['btc_moving_avg_30d'] = df[price_col].rolling(window=30).mean()
    df['btc_moving_avg_200d'] = df[price_col].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    df['btc_rsi_14'] = ta.momentum.RSIIndicator(close=df[price_col], window=14).rsi()
    
    # MACD and MACD signal
    macd = ta.trend.MACD(close=df[price_col])
    df['btc_macd'] = macd.macd()
    df['btc_macd_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    boll = ta.volatility.BollingerBands(close=df[price_col], window=20, window_dev=2)
    df['btc_bollinger_upper'] = boll.bollinger_hband()
    df['btc_bollinger_lower'] = boll.bollinger_lband()
    df['btc_bollinger_bandwidth'] = df['btc_bollinger_upper'] - df['btc_bollinger_lower']
    
    return df



def plot_btc_price_with_moving_averages(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['cryptos_BTC'], label='BTC Price', linewidth=2)
    plt.plot(df.index, df['btc_moving_avg_30d'], label='30D MA', linestyle='--')
    plt.plot(df.index, df['btc_moving_avg_200d'], label='200D MA', linestyle='--')
    plt.title('BTC Price with 30D and 200D Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_btc_rsi(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['btc_rsi_14'], label='RSI 14D', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('BTC RSI (Relative Strength Index)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_btc_macd(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['btc_macd'], label='MACD', color='blue')
    plt.plot(df.index, df['btc_macd_signal'], label='Signal Line', color='orange')
    plt.title('BTC MACD and Signal Line')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_btc_bollinger_bands(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['cryptos_BTC'], label='BTC Price', color='black')
    plt.plot(df.index, df['btc_bollinger_upper'], label='Upper Band', color='green', linestyle='--')
    plt.plot(df.index, df['btc_bollinger_lower'], label='Lower Band', color='red', linestyle='--')
    plt.fill_between(df.index, df['btc_bollinger_lower'], df['btc_bollinger_upper'], color='gray', alpha=0.2)
    plt.title('BTC Price with Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


df = pd.read_csv('data_with_seasonality.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

df = add_btc_technical_indicators(df)
plot_btc_price_with_moving_averages(df)
plot_btc_rsi(df)
plot_btc_macd(df)
plot_btc_bollinger_bands(df)

df.to_csv("data_with_technical_indicators.csv")

print("="*60)
print(df.columns.values)
print(len(df.columns.values))
print("="*60)