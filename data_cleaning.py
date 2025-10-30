import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from typing import Optional, List, Dict, Any


class BusinessDaysProcessor:
    """
    A class for processing financial data to include only business days and adding
    comprehensive holiday and temporal features for machine learning models.
    
    This class handles US market holidays, weekend filtering, and creates various
    features related to trading day patterns, holiday effects, and market regimes.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the Business Days Processor.
        
        Args:
            verbose: Whether to print detailed processing information
        """
        self.verbose = verbose
        self.holidays: Optional[pd.DatetimeIndex] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.feature_summary: Dict[str, Any] = {}
        self.processing_log: List[str] = []
    
    def _log(self, message: str) -> None:
        """Log a message and optionally print it."""
        self.processing_log.append(message)
        if self.verbose:
            print(message)
    
    def get_us_holidays(self, start_year: int = 2020, end_year: int = 2025) -> pd.DatetimeIndex:
        """
        Get major US market holidays that affect trading.
        Includes all NYSE/NASDAQ market closures and early closes.
        
        Args:
            start_year: Starting year for holiday calculation
            end_year: Ending year for holiday calculation
            
        Returns:
            DatetimeIndex containing all holiday dates
        """
        holidays = []
        
        def add_with_weekend_rule(date, name):
            """Add holiday with weekend observance rules"""
            if date.weekday() == 5:  # Saturday
                observed = date - pd.Timedelta(days=1)  # Friday
                holidays.append(observed.strftime("%Y-%m-%d"))
                self._log(f"{name} {date.strftime('%Y-%m-%d')} observed on {observed.strftime('%Y-%m-%d')} (Friday)")
            elif date.weekday() == 6:  # Sunday
                observed = date + pd.Timedelta(days=1)  # Monday
                holidays.append(observed.strftime("%Y-%m-%d"))
                self._log(f"{name} {date.strftime('%Y-%m-%d')} observed on {observed.strftime('%Y-%m-%d')} (Monday)")
            else:
                holidays.append(date.strftime("%Y-%m-%d"))
        
        def easter_date(year):
            """Calculate Easter Sunday for given year (Western/Gregorian calendar)"""
            # Using the algorithm for Easter calculation
            a = year % 19
            b = year // 100
            c = year % 100
            d = b // 4
            e = b % 4
            f = (b + 8) // 25
            g = (b - f + 1) // 3
            h = (19 * a + b - d - g + 15) % 30
            i = c // 4
            k = c % 4
            l = (32 + 2 * e + 2 * i - h - k) % 7
            m = (a + 11 * h + 22 * l) // 451
            n = (h + l - 7 * m + 114) // 31
            p = (h + l - 7 * m + 114) % 31
            return pd.Timestamp(year, n, p + 1)
        
        for year in range(start_year, end_year + 1):
            # New Year's Day
            new_years = pd.Timestamp(f"{year}-01-01")
            add_with_weekend_rule(new_years, "New Year's Day")
            
            # Martin Luther King Jr. Day (3rd Monday in January)
            jan_first = pd.Timestamp(f"{year}-01-01")
            # Find first Monday, then add 14 days to get 3rd Monday
            first_monday = jan_first + pd.Timedelta(days=(7 - jan_first.weekday()) % 7)
            mlk_day = first_monday + pd.Timedelta(days=14)
            holidays.append(mlk_day.strftime("%Y-%m-%d"))
            
            # Presidents Day (3rd Monday in February)
            feb_first = pd.Timestamp(f"{year}-02-01")
            first_monday = feb_first + pd.Timedelta(days=(7 - feb_first.weekday()) % 7)
            presidents_day = first_monday + pd.Timedelta(days=14)
            holidays.append(presidents_day.strftime("%Y-%m-%d"))
            
            # Good Friday (Friday before Easter)
            easter = easter_date(year)
            good_friday = easter - pd.Timedelta(days=2)
            holidays.append(good_friday.strftime("%Y-%m-%d"))
            
            # Memorial Day (Last Monday in May)
            may_31 = pd.Timestamp(f"{year}-05-31")
            # Go back to find the last Monday
            memorial_day = may_31 - pd.Timedelta(days=(may_31.weekday() + 1) % 7)
            holidays.append(memorial_day.strftime("%Y-%m-%d"))
            
            # Juneteenth (June 19) - became federal holiday in 2021
            if year >= 2021:
                juneteenth = pd.Timestamp(f"{year}-06-19")
                add_with_weekend_rule(juneteenth, "Juneteenth")
            
            # Independence Day (July 4)
            independence = pd.Timestamp(f"{year}-07-04")
            add_with_weekend_rule(independence, "Independence Day")
            
            # Labor Day (1st Monday in September)
            sep_first = pd.Timestamp(f"{year}-09-01")
            # Find first Monday
            labor_day = sep_first + pd.Timedelta(days=(7 - sep_first.weekday()) % 7)
            holidays.append(labor_day.strftime("%Y-%m-%d"))
            
            # Thanksgiving (4th Thursday in November)
            nov_first = pd.Timestamp(f"{year}-11-01")
            # Find first Thursday, then add 21 days to get 4th Thursday
            first_thursday = nov_first + pd.Timedelta(days=(3 - nov_first.weekday()) % 7)
            thanksgiving = first_thursday + pd.Timedelta(days=21)
            holidays.append(thanksgiving.strftime("%Y-%m-%d"))
            
            # Christmas Day (December 25)
            christmas = pd.Timestamp(f"{year}-12-25")
            add_with_weekend_rule(christmas, "Christmas Day")
            
            # Christmas Eve (December 24) - markets often close early or fully
            christmas_eve = pd.Timestamp(f"{year}-12-24")
            if christmas_eve.weekday() < 5:  # Only if it's a weekday
                holidays.append(christmas_eve.strftime("%Y-%m-%d"))
        
        # Add special one-time holidays
        special_holidays = {
            "2025-01-09": "National Day of Mourning"  # Example special closure
        }
        
        for date_str, name in special_holidays.items():
            if start_year <= int(date_str[:4]) <= end_year:
                holidays.append(date_str)
                self._log(f"Added special holiday: {name} on {date_str}")
        
        # Remove duplicates and sort
        holidays = sorted(list(set(holidays)))
        self.holidays = pd.to_datetime(holidays)
        
        self._log(f"Generated {len(self.holidays)} holidays for years {start_year}-{end_year}")
        return self.holidays
    def _add_basic_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic temporal features like day of week and gap analysis.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with basic temporal features added
        """
        # Add day-of-week features for Friday effect, Monday effect
        df['day_of_week'] = df.index.dayofweek
        df['is_monday'] = (df.index.dayofweek == 0).astype(int)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)
        
        # Add "days since last trading day" feature
        df['days_gap'] = df.index.to_series().diff().dt.days
        df['is_after_weekend'] = (df['days_gap'] > 1).astype(int)
        
        self._log("Added basic temporal features: day_of_week, is_monday, is_friday, days_gap, is_after_weekend")
        return df
    
    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive holiday-related features.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with holiday features added
        """
        if self.holidays is None:
            start_year = df.index.year.min()
            end_year = df.index.year.max()
            self.get_us_holidays(start_year, end_year)
        
        # Basic holiday features
        df['is_holiday'] = df.index.isin(self.holidays).astype(int)
        df['is_pre_holiday'] = df.index.isin(self.holidays - pd.Timedelta(days=1)).astype(int)
        df['is_post_holiday'] = df.index.isin(self.holidays + pd.Timedelta(days=1)).astype(int)
        
        # Holiday week features
        df['is_holiday_week'] = 0
        for holiday in self.holidays:
            # Week containing holiday
            week_start = holiday - pd.Timedelta(days=holiday.weekday())
            week_end = week_start + pd.Timedelta(days=6)
            mask = (df.index >= week_start) & (df.index <= week_end)
            df.loc[mask, 'is_holiday_week'] = 1
        
        # Days to/from next/last holiday
        df['days_to_next_holiday'] = 999
        df['days_from_last_holiday'] = 999
        
        for i, date in enumerate(df.index):
            # Days to next holiday
            future_holidays = self.holidays[self.holidays > date]
            if len(future_holidays) > 0:
                df.iloc[i, df.columns.get_loc('days_to_next_holiday')] = (future_holidays[0] - date).days
            
            # Days from last holiday
            past_holidays = self.holidays[self.holidays < date]
            if len(past_holidays) > 0:
                df.iloc[i, df.columns.get_loc('days_from_last_holiday')] = (date - past_holidays[-1]).days
        
        # **NEW: Specific holiday type features**
        self._add_specific_holiday_features(df)
        
        self._log("Added holiday features: is_holiday, is_pre_holiday, is_post_holiday, is_holiday_week, days_to/from_holiday, specific holiday types")
        return df

    def _add_specific_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features for specific types of holidays with different market impacts.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with specific holiday features added
        """
        start_year = df.index.year.min()
        end_year = df.index.year.max()
        
        # Initialize specific holiday features
        df['is_memorial_day_week'] = 0
        df['is_independence_day_week'] = 0
        df['is_labor_day_week'] = 0
        df['is_mlk_day_week'] = 0
        df['is_presidents_day_week'] = 0
        df['is_good_friday_week'] = 0
        df['is_juneteenth_week'] = 0
        
        # Long weekend indicators (3+ day weekends)
        df['is_long_weekend_start'] = 0  # Friday before long weekend
        df['is_long_weekend_end'] = 0    # Tuesday after long weekend
        
        for year in range(start_year, end_year + 1):
            # Memorial Day (Last Monday in May) - unofficial start of summer
            may_31 = pd.Timestamp(f"{year}-05-31")
            memorial_day = may_31 - pd.Timedelta(days=(may_31.weekday() + 1) % 7)
            
            # Memorial Day week features
            memorial_week_start = memorial_day - pd.Timedelta(days=memorial_day.weekday())
            memorial_week_end = memorial_week_start + pd.Timedelta(days=6)
            mask = (df.index >= memorial_week_start) & (df.index <= memorial_week_end)
            df.loc[mask, 'is_memorial_day_week'] = 1
            
            # Memorial Day creates a long weekend - mark Friday before and Tuesday after
            memorial_friday = memorial_day - pd.Timedelta(days=3)  # Friday before Monday
            memorial_tuesday = memorial_day + pd.Timedelta(days=1)  # Tuesday after Monday
            
            if memorial_friday in df.index:
                df.loc[memorial_friday, 'is_long_weekend_start'] = 1
            if memorial_tuesday in df.index:
                df.loc[memorial_tuesday, 'is_long_weekend_end'] = 1
            
            # Independence Day (July 4) week
            independence = pd.Timestamp(f"{year}-07-04")
            july4_week_start = independence - pd.Timedelta(days=independence.weekday())
            july4_week_end = july4_week_start + pd.Timedelta(days=6)
            mask = (df.index >= july4_week_start) & (df.index <= july4_week_end)
            df.loc[mask, 'is_independence_day_week'] = 1
            
            # Labor Day (1st Monday in September) - end of summer
            sep_first = pd.Timestamp(f"{year}-09-01")
            labor_day = sep_first + pd.Timedelta(days=(7 - sep_first.weekday()) % 7)
            
            labor_week_start = labor_day - pd.Timedelta(days=labor_day.weekday())
            labor_week_end = labor_week_start + pd.Timedelta(days=6)
            mask = (df.index >= labor_week_start) & (df.index <= labor_week_end)
            df.loc[mask, 'is_labor_day_week'] = 1
            
            # Labor Day also creates long weekend
            labor_friday = labor_day - pd.Timedelta(days=3)
            labor_tuesday = labor_day + pd.Timedelta(days=1)
            
            if labor_friday in df.index:
                df.loc[labor_friday, 'is_long_weekend_start'] = 1
            if labor_tuesday in df.index:
                df.loc[labor_tuesday, 'is_long_weekend_end'] = 1
            
            # MLK Day (3rd Monday in January)
            jan_first = pd.Timestamp(f"{year}-01-01")
            first_monday = jan_first + pd.Timedelta(days=(7 - jan_first.weekday()) % 7)
            mlk_day = first_monday + pd.Timedelta(days=14)
            
            mlk_week_start = mlk_day - pd.Timedelta(days=mlk_day.weekday())
            mlk_week_end = mlk_week_start + pd.Timedelta(days=6)
            mask = (df.index >= mlk_week_start) & (df.index <= mlk_week_end)
            df.loc[mask, 'is_mlk_day_week'] = 1
            
            # Presidents Day (3rd Monday in February)
            feb_first = pd.Timestamp(f"{year}-02-01")
            first_monday = feb_first + pd.Timedelta(days=(7 - feb_first.weekday()) % 7)
            presidents_day = first_monday + pd.Timedelta(days=14)
            
            pres_week_start = presidents_day - pd.Timedelta(days=presidents_day.weekday())
            pres_week_end = pres_week_start + pd.Timedelta(days=6)
            mask = (df.index >= pres_week_start) & (df.index <= pres_week_end)
            df.loc[mask, 'is_presidents_day_week'] = 1
            
            # Good Friday (varies each year)
            def easter_date(year):
                """Calculate Easter Sunday for given year"""
                a = year % 19
                b = year // 100
                c = year % 100
                d = b // 4
                e = b % 4
                f = (b + 8) // 25
                g = (b - f + 1) // 3
                h = (19 * a + b - d - g + 15) % 30
                i = c // 4
                k = c % 4
                l = (32 + 2 * e + 2 * i - h - k) % 7
                m = (a + 11 * h + 22 * l) // 451
                n = (h + l - 7 * m + 114) // 31
                p = (h + l - 7 * m + 114) % 31
                return pd.Timestamp(year, n, p + 1)
            
            easter = easter_date(year)
            good_friday = easter - pd.Timedelta(days=2)
            
            gf_week_start = good_friday - pd.Timedelta(days=good_friday.weekday())
            gf_week_end = gf_week_start + pd.Timedelta(days=6)
            mask = (df.index >= gf_week_start) & (df.index <= gf_week_end)
            df.loc[mask, 'is_good_friday_week'] = 1
            
            # Juneteenth (June 19) - since 2021
            if year >= 2021:
                juneteenth = pd.Timestamp(f"{year}-06-19")
                jt_week_start = juneteenth - pd.Timedelta(days=juneteenth.weekday())
                jt_week_end = jt_week_start + pd.Timedelta(days=6)
                mask = (df.index >= jt_week_start) & (df.index <= jt_week_end)
                df.loc[mask, 'is_juneteenth_week'] = 1
        
        # Summer trading period (Memorial Day to Labor Day - typically lower volume)
        df['is_summer_trading'] = 0
        for year in range(start_year, end_year + 1):
            # Memorial Day to Labor Day
            may_31 = pd.Timestamp(f"{year}-05-31")
            memorial_day = may_31 - pd.Timedelta(days=(may_31.weekday() + 1) % 7)
            
            sep_first = pd.Timestamp(f"{year}-09-01")
            labor_day = sep_first + pd.Timedelta(days=(7 - sep_first.weekday()) % 7)
            
            mask = (df.index >= memorial_day) & (df.index <= labor_day)
            df.loc[mask, 'is_summer_trading'] = 1
        
        self._log("Added specific holiday features: memorial_day_week, independence_day_week, labor_day_week, etc.")
        return df

    def _add_special_period_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features for special market periods like Christmas week, New Year, etc.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with special period features added
        """
        start_year = df.index.year.min()
        end_year = df.index.year.max()
        
        # Initialize special period features
        df['is_christmas_week'] = 0
        df['is_new_year_week'] = 0
        df['is_thanksgiving_week'] = 0
        df['is_holiday_season'] = 0  # Thanksgiving to New Year
        df['is_january_effect'] = 0  # First few days of January
        df['is_december_effect'] = 0  # Last few days of December
        
        for year in range(start_year, end_year + 1):
            # Christmas week (often low volume)
            christmas = pd.Timestamp(f"{year}-12-25")
            xmas_week_start = christmas - pd.Timedelta(days=7)
            xmas_week_end = christmas + pd.Timedelta(days=7)
            mask = (df.index >= xmas_week_start) & (df.index <= xmas_week_end)
            df.loc[mask, 'is_christmas_week'] = 1
            
            # New Year week
            new_year = pd.Timestamp(f"{year}-01-01")
            ny_week_start = new_year - pd.Timedelta(days=7)
            ny_week_end = new_year + pd.Timedelta(days=7)
            mask = (df.index >= ny_week_start) & (df.index <= ny_week_end)
            df.loc[mask, 'is_new_year_week'] = 1
            
            # Thanksgiving week (short week, often volatile)
            nov_first = pd.Timestamp(f"{year}-11-01")
            first_thursday = nov_first + pd.Timedelta(days=(3 - nov_first.weekday()) % 7)
            thanksgiving = first_thursday + pd.Timedelta(days=21)
            tg_week_start = thanksgiving - pd.Timedelta(days=3)
            tg_week_end = thanksgiving + pd.Timedelta(days=3)
            mask = (df.index >= tg_week_start) & (df.index <= tg_week_end)
            df.loc[mask, 'is_thanksgiving_week'] = 1
            
            # Holiday season (Thanksgiving to New Year) - typically lower volume
            holiday_season_start = thanksgiving
            holiday_season_end = pd.Timestamp(f"{year+1}-01-02") if year < end_year else pd.Timestamp(f"{year}-12-31")
            mask = (df.index >= holiday_season_start) & (df.index <= holiday_season_end)
            df.loc[mask, 'is_holiday_season'] = 1
            
            # January Effect (first 5 trading days of January)
            jan_start = pd.Timestamp(f"{year}-01-01")
            jan_effect_end = pd.Timestamp(f"{year}-01-10")  # Conservative end date
            mask = (df.index >= jan_start) & (df.index <= jan_effect_end)
            df.loc[mask, 'is_january_effect'] = 1
            
            # December Effect (last 5 trading days of December)
            dec_effect_start = pd.Timestamp(f"{year}-12-20")
            dec_end = pd.Timestamp(f"{year}-12-31")
            mask = (df.index >= dec_effect_start) & (df.index <= dec_end)
            df.loc[mask, 'is_december_effect'] = 1
        
        self._log("Added special period features: christmas_week, new_year_week, thanksgiving_week, holiday_season, january/december_effects")
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime and enhanced features.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with market regime features added
        """
        # Enhanced gap analysis (accounts for holidays)
        df['is_after_long_break'] = ((df['days_gap'] > 3) | 
                                     (df['is_post_holiday'] == 1)).astype(int)
        
        # Market regime indicators
        df['is_low_volume_period'] = ((df['is_christmas_week'] == 1) | 
                                      (df['is_new_year_week'] == 1)).astype(int)
        
        self._log("Added market regime features: is_after_long_break, is_low_volume_period")
        return df
    
    def prepare_business_days_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to process data: keep only business days and create ML features.
        
        Args:
            df: Input DataFrame with any datetime index
            
        Returns:
            Processed DataFrame with only business days and comprehensive features
        """
        self._log("Starting business days data preparation...")
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        # Keep only business days (removes weekends automatically)
        df_business = df[df.index.to_series().dt.dayofweek < 5].copy()
        self._log(f"Filtered to weekdays: {len(df)} -> {len(df_business)} rows")
        
        # Get holidays for the date range
        start_year = df_business.index.year.min()
        end_year = df_business.index.year.max()
        self.get_us_holidays(start_year, end_year)
        
        # **THIS IS THE MISSING STEP** - Remove holidays from the dataset
        holidays_in_data = df_business.index.intersection(self.holidays)
        if len(holidays_in_data) > 0:
            self._log(f"Found {len(holidays_in_data)} holidays in data to remove:")
            for holiday in holidays_in_data:
                self._log(f"  Removing holiday: {holiday.strftime('%Y-%m-%d (%A)')}")
            
            # Filter out holidays
            df_business = df_business[~df_business.index.isin(self.holidays)]
            self._log(f"After removing holidays: {len(df_business)} rows remaining")
        else:
            self._log("No holidays found in the data range")
        
        # Add all feature categories (now on the holiday-filtered data)
        df_business = self._add_basic_temporal_features(df_business)
        df_business = self._add_holiday_features(df_business)
        df_business = self._add_special_period_features(df_business)
        df_business = self._add_market_regime_features(df_business)
        
        # Store processed data
        self.processed_df = df_business
        
        # Generate summary statistics
        self._generate_feature_summary()
        
        self._log(f"Business days processing complete. Final dataset: {len(df_business)} trading days")
        return df_business
    
    def _generate_feature_summary(self) -> None:
        """Generate summary statistics about the processed data."""
        if self.processed_df is None:
            return
        
        df = self.processed_df
        
        self.feature_summary = {
            'total_trading_days': len(df),
            'holiday_periods': {
                'holidays': df['is_holiday'].sum(),
                'pre_holidays': df['is_pre_holiday'].sum(),
                'post_holidays': df['is_post_holiday'].sum(),
                'holiday_weeks': df['is_holiday_week'].sum()
            },
            'special_periods': {
                'christmas_week_days': df['is_christmas_week'].sum(),
                'new_year_week_days': df['is_new_year_week'].sum(),
                'thanksgiving_week_days': df['is_thanksgiving_week'].sum(),
                'low_volume_periods': df['is_low_volume_period'].sum()
            },
            'temporal_patterns': {
                'mondays': df['is_monday'].sum(),
                'fridays': df['is_friday'].sum(),
                'after_long_breaks': df['is_after_long_break'].sum(),
                'after_weekends': df['is_after_weekend'].sum()
            },
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d'),
                'end': df.index.max().strftime('%Y-%m-%d')
            }
        }
        
        # Log summary
        self._log("Business days dataset prepared:")
        self._log(f"  Total trading days: {self.feature_summary['total_trading_days']}")
        self._log(f"  Holiday periods: {self.feature_summary['holiday_periods']['holidays']} holidays, "
                 f"{self.feature_summary['holiday_periods']['pre_holidays']} pre-holidays, "
                 f"{self.feature_summary['holiday_periods']['post_holidays']} post-holidays")
        self._log(f"  Special periods: {self.feature_summary['special_periods']['christmas_week_days']} Christmas week days")
        self._log(f"  Date range: {self.feature_summary['date_range']['start']} to {self.feature_summary['date_range']['end']}")
    
    def get_feature_list(self) -> List[str]:
        """
        Get list of all added features.
        
        Returns:
            List of feature column names
        """
        if self.processed_df is None:
            return []
        
        feature_keywords = ['day_', 'is_', 'days_', 'holiday']
        feature_cols = [col for col in self.processed_df.columns 
                       if any(keyword in col.lower() for keyword in feature_keywords)]
        return feature_cols
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the processing results.
        
        Returns:
            Dictionary containing summary statistics
        """
        return self.feature_summary.copy()
    
    def analyze_missing_data(self) -> pd.Series:
        """
        Analyze missing data in the processed dataset.
        
        Returns:
            Series with missing value counts by column
        """
        if self.processed_df is None:
            self._log("No processed data available. Run prepare_business_days_data() first.")
            return pd.Series()
        
        missing_data = self.processed_df.isnull().sum().sort_values(ascending=False)
        
        if self.verbose and missing_data.sum() > 0:
            self._log(f"\nMissing data analysis:")
            self._log(f"Total missing values: {missing_data.sum():,}")
            self._log(f"Columns with missing data:")
            for col, count in missing_data[missing_data > 0].head(10).items():
                pct = count / len(self.processed_df) * 100
                self._log(f"  {col}: {count:,} ({pct:.1f}%)")
        
        return missing_data
    
    def find_missing_dates(self, column_name: str) -> pd.DatetimeIndex:
        """
        Find dates where a specific column has missing values.
        
        Args:
            column_name: Name of the column to check
            
        Returns:
            DatetimeIndex of dates with missing values
        """
        if self.processed_df is None or column_name not in self.processed_df.columns:
            return pd.DatetimeIndex([])
        
        missing_dates = self.processed_df[self.processed_df[column_name].isnull()].index
        
        if self.verbose and len(missing_dates) > 0:
            self._log(f"Missing dates for '{column_name}': {len(missing_dates)} dates")
            if len(missing_dates) <= 10:
                for date in missing_dates:
                    self._log(f"  {date.strftime('%Y-%m-%d')}")
            else:
                self._log(f"  First 5: {missing_dates[:5].strftime('%Y-%m-%d').tolist()}")
                self._log(f"  Last 5: {missing_dates[-5:].strftime('%Y-%m-%d').tolist()}")
        
        return missing_dates
    
    def save_processed_data(self, output_path: str = "clean_data.csv") -> bool:
        """
        Save the processed DataFrame to a CSV file.
        
        Args:
            output_path: Path where to save the processed data
            
        Returns:
            True if successful, False otherwise
        """
        if self.processed_df is None:
            self._log("No processed data to save. Run prepare_business_days_data() first.")
            return False
        
        try:
            self.processed_df.to_csv(output_path)
            self._log(f"Processed data saved to {output_path}")
            return True
        except Exception as e:
            self._log(f"Error saving file: {e}")
            return False
    
    def get_processing_log(self) -> List[str]:
        """
        Get the complete processing log.
        
        Returns:
            List of log messages
        """
        return self.processing_log.copy()


# Usage example and test
if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'btc_price': np.random.randn(len(dates)) * 1000 + 50000,
        'tech_sector': np.random.randn(len(dates)) * 10 + 100
    }, index=dates)
    
    # Create processor and process the data
    processor = BusinessDaysProcessor(verbose=True)
    processed_data = processor.prepare_business_days_data(sample_data)
    
    print("\nFeatures added:")
    feature_cols = processor.get_feature_list()
    for col in feature_cols:
        print(f"  {col}")
    
    print(f"\nSample of processed data:")
    print(processed_data[feature_cols].head(10))
    
    # Show summary
    summary = processor.get_summary()
    print(f"\nProcessing Summary:")
    print(f"  Total trading days: {summary['total_trading_days']}")
    print(f"  Holiday effects: {summary['holiday_periods']}")
    
    # Example usage with real merged data (commented out)
    """
    # Usage with actual merged data
    merged_df = pd.read_csv("merged_data.csv", index_col='Date')
    processor = BusinessDaysProcessor(verbose=True)
    df_clean = processor.prepare_business_days_data(merged_df)
    
    # Analyze the results
    print(df_clean.info())
    missing_data = processor.analyze_missing_data()
    
    # Check specific columns for missing dates
    missing_gold = processor.find_missing_dates('commodities_SPDR Gold Shares')
    missing_industrials = processor.find_missing_dates('key_sectors_Industrials')
    
    # Save the processed data
    processor.save_processed_data("clean_data.csv")
    """