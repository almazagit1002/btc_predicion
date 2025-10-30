import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_clean_data(file_path="clean_data.csv"):
    """
    Load the clean data CSV file with proper date parsing
    """
    try:
        # Try reading with first column as date index
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def add_seasonality_features(df):
    """
    Add comprehensive seasonality features to the dataset
    """
    df = df.copy()
    
    print("Adding seasonality features...")
    
    # === BASIC TIME COMPONENTS ===
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_month'] = df.index.day
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    
    # === CYCLICAL ENCODING ===
    # (Preserves circular nature - January connects to December)
    
    # Month cyclical (12 months)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Quarter cyclical (4 quarters)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Day of month cyclical (31 days)
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    
    # Day of year cyclical (365 days)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Week of year cyclical (52 weeks)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Day of week cyclical (7 days)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # === MARKET-SPECIFIC SEASONAL PATTERNS ===
    
    # Monthly effects
    df['is_january'] = (df['month'] == 1).astype(int)    # January effect (small caps rally)
    df['is_february'] = (df['month'] == 2).astype(int)   # February correction often
    df['is_march'] = (df['month'] == 3).astype(int)      # Q1 end, rebalancing
    df['is_april'] = (df['month'] == 4).astype(int)      # Tax day effects
    df['is_may'] = (df['month'] == 5).astype(int)        # "Sell in May" effect
    df['is_october'] = (df['month'] == 10).astype(int)   # October crashes historically
    df['is_november'] = (df['month'] == 11).astype(int)  # Election month (every 2 years)
    df['is_december'] = (df['month'] == 12).astype(int)  # Santa rally
    
    # Quarterly effects
    df['is_q1'] = (df['quarter'] == 1).astype(int)  # Q1 earnings, new year optimism
    df['is_q2'] = (df['quarter'] == 2).astype(int)  # Summer doldrums start
    df['is_q3'] = (df['quarter'] == 3).astype(int)  # Summer doldrums
    df['is_q4'] = (df['quarter'] == 4).astype(int)  # Year-end effects, Santa rally
    
    # Day of week effects
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)     # Monday blues
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)    # Tuesday often strong
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)  # Mid-week
    df['is_thursday'] = (df['day_of_week'] == 3).astype(int)   # Thursday
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)     # Friday effects
    
    # === EARNINGS SEASONS ===
    # Earnings typically announced in: January, April, July, October
    df['is_earnings_season'] = df['month'].isin([1, 4, 7, 10]).astype(int)
    df['is_earnings_month'] = ((df['month'].isin([1, 4, 7, 10])) & 
                              (df['day_of_month'] <= 20)).astype(int)
    
    # === CRYPTO-SPECIFIC SEASONALITY ===
    # Bitcoin halving cycle (approximately every 4 years)
    # Next halvings: 2024, 2028, 2032...
    df['years_since_2020'] = df['year'] - 2020
    df['halving_cycle_position'] = (df['years_since_2020'] % 4) / 4
    df['is_halving_year'] = ((df['year'] - 2020) % 4 == 0).astype(int)
    df['is_pre_halving_year'] = ((df['year'] - 2019) % 4 == 0).astype(int)
    df['is_post_halving_year'] = ((df['year'] - 2021) % 4 == 0).astype(int)
    
    # === MACRO ECONOMIC CALENDAR ===
    # Federal Reserve meeting months (8 times per year)
    fed_months = [1, 3, 5, 6, 7, 9, 11, 12]  # Typical FOMC schedule
    df['is_fed_month'] = df['month'].isin(fed_months).astype(int)
    
    # Quarter-end effects (rebalancing)
    df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
    df['is_year_end'] = (df['month'] == 12).astype(int)
    df['is_quarter_start'] = df['month'].isin([1, 4, 7, 10]).astype(int)
    
    # Month-end effects (last 5 trading days of month)
    df['days_from_month_end'] = (df.index.to_series().dt.day - 
                                df.index.to_series().dt.days_in_month).abs()
    df['is_month_end'] = (df['days_from_month_end'] <= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    
    # === SEASONAL VOLATILITY INDICATORS ===
    # Summer doldrums (May-September: typically lower volatility)
    df['is_summer_doldrums'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    
    # High volatility periods
    df['is_high_vol_season'] = df['month'].isin([1, 2, 3, 10, 11, 12]).astype(int)
    
    # === TAX-RELATED SEASONALITY ===
    # US tax season effects
    df['is_tax_season'] = df['month'].isin([1, 2, 3, 4]).astype(int)
    df['is_tax_deadline_month'] = (df['month'] == 4).astype(int)
    
    # Tax loss selling season
    df['is_tax_loss_season'] = df['month'].isin([11, 12]).astype(int)
    
    print(f"✓ Added {len([col for col in df.columns if any(x in col for x in ['month', 'quarter', 'day_', 'week_', 'is_', 'season'])])} seasonality features")
    
    return df

def analyze_seasonality_patterns(df, target_col='cryptos_BTC'):
    """
    Analyze seasonal patterns in the target variable
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found. Available columns:")
        print([col for col in df.columns if 'crypto' in col.lower() or 'btc' in col.lower()])
        return None
    
    print(f"\nAnalyzing seasonality patterns for {target_col}...")
    
    # Calculate returns
    df['returns'] = df[target_col].pct_change()
    
    # Monthly analysis
    monthly_stats = df.groupby('month')['returns'].agg(['mean', 'std', 'count']).round(4)
    monthly_stats['annualized_return'] = monthly_stats['mean'] * 252
    monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print("\nMonthly Return Statistics:")
    print(monthly_stats)
    
    # Quarterly analysis
    quarterly_stats = df.groupby('quarter')['returns'].agg(['mean', 'std', 'count']).round(4)
    quarterly_stats['annualized_return'] = quarterly_stats['mean'] * 252
    quarterly_stats.index = ['Q1', 'Q2', 'Q3', 'Q4']
    
    print("\nQuarterly Return Statistics:")
    print(quarterly_stats)
    
    # Day of week analysis
    dow_stats = df.groupby('day_of_week')['returns'].agg(['mean', 'std', 'count']).round(4)
    dow_stats['annualized_return'] = dow_stats['mean'] * 252
    dow_stats.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    print("\nDay of Week Return Statistics:")
    print(dow_stats)
    
    return {
        'monthly': monthly_stats,
        'quarterly': quarterly_stats,
        'day_of_week': dow_stats
    }

def plot_seasonality_patterns(df, target_col='cryptos_BTC', save_plots=True):
    """
    Create visualizations of seasonal patterns
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return
    
    # Calculate returns
    df['returns'] = df[target_col].pct_change()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Seasonality Analysis for {target_col}', fontsize=16)
    
    # Monthly returns
    monthly_returns = df.groupby('month')['returns'].mean()
    monthly_returns.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns.plot(kind='bar', ax=axes[0,0], color='steelblue')
    axes[0,0].set_title('Average Monthly Returns')
    axes[0,0].set_ylabel('Return')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Quarterly returns
    quarterly_returns = df.groupby('quarter')['returns'].mean()
    quarterly_returns.index = ['Q1', 'Q2', 'Q3', 'Q4']
    quarterly_returns.plot(kind='bar', ax=axes[0,1], color='forestgreen')
    axes[0,1].set_title('Average Quarterly Returns')
    axes[0,1].set_ylabel('Return')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # Day of week returns
    dow_returns = df.groupby('day_of_week')['returns'].mean()
    dow_returns.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    dow_returns.plot(kind='bar', ax=axes[1,0], color='darkorange')
    axes[1,0].set_title('Average Day-of-Week Returns')
    axes[1,0].set_ylabel('Return')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Monthly volatility (standard deviation)
    monthly_vol = df.groupby('month')['returns'].std()
    monthly_vol.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_vol.plot(kind='bar', ax=axes[1,1], color='crimson')
    axes[1,1].set_title('Monthly Volatility (Std Dev)')
    axes[1,1].set_ylabel('Volatility')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('seasonality_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved seasonality analysis plot as 'seasonality_analysis.png'")
    
    plt.show()

def save_data_with_seasonality(df, output_file="data_with_seasonality.csv"):
    """
    Save the dataframe with seasonality features
    """
    try:
        df.to_csv(output_file)
        print(f"✓ Saved data with seasonality features to '{output_file}'")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show summary of added features
        seasonality_cols = [col for col in df.columns if any(x in col for x in 
                           ['month', 'quarter', 'day_', 'week_', 'is_', 'season', 'sin', 'cos'])]
        
        print(f"  Seasonality features added: {len(seasonality_cols)}")
        
    except Exception as e:
        print(f"Error saving file: {e}")
def filter_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encoded and relevant time-based features to keep
    keep_time_features = [
        # Days of week (if you're not predicting on weekends, remove those)
        'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 'is_friday',

        # Months
        'is_january', 'is_february', 'is_march', 'is_april', 'is_may',
        'is_october', 'is_november', 'is_december',

        # Quarters
        'is_q1', 'is_q2', 'is_q3', 'is_q4',

        # Other useful time flags
        'is_month_end', 'is_month_start',
        'is_christmas_week', 'is_new_year_week', 'is_thanksgiving_week',
        'is_earnings_season', 'is_earnings_month',
        'is_tax_season', 'is_tax_deadline_month', 'is_tax_loss_season'
    ]

    # Identify all time-related columns to potentially drop
    drop_patterns = [
        'day_of_week', 'month', 'quarter', 'week_of_year',
        'day_of_month', 'day_of_year', 'week',
        '_sin', '_cos',  # Drop all cyclical encodings
    ]

    # Drop matching columns
    drop_columns = [
        col for col in df.columns
        if any(p in col for p in drop_patterns)
        and col not in keep_time_features
    ]

    df_filtered = df.drop(columns=drop_columns, errors='ignore')
    
    return df_filtered
def main():
    """
    Main function to process seasonality features
    """
    print("="*60)
    print("SEASONALITY FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    df = load_clean_data("clean_data.csv")
    if df is None:
        return
    
    # Add seasonality features
    df_with_seasonality = add_seasonality_features(df)
 
    
    # Analyze patterns (find BTC column)
    btc_cols = [col for col in df.columns if 'btc' in col.lower()]
    if btc_cols:
        target_col = btc_cols[0]
        print(f"\nUsing '{target_col}' for seasonality analysis")
        
        # Analyze patterns
        patterns = analyze_seasonality_patterns(df_with_seasonality, target_col)
        
        # Create plots
        plot_seasonality_patterns(df_with_seasonality, target_col)
        
    else:
        print("No BTC column found for analysis")
    
    df_filtered = filter_time_features(df_with_seasonality)
    # Save enhanced data
    save_data_with_seasonality(df_filtered)
    
    print("\n" + "="*60)
    print("SEASONALITY FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(df_with_seasonality.columns.values)
    print(len(df_with_seasonality.columns.values))
    print("="*60)
    print(df_filtered.columns.values)
    print(len(df_filtered.columns.values))
    
    
    

if __name__ == "__main__":
    main()