import pandas as pd
from pathlib import Path
from typing import Optional, List, Union


class CSVMerger:
    """
    A class for merging multiple CSV files from a directory into a single DataFrame.
    
    This class handles date detection, column naming, missing values reporting,
    and provides various merge strategies for combining time series data.
    """
    
    def __init__(self, data_dir: str = "data", verbose: bool = True):
        """
        Initialize the CSV Merger.
        
        Args:
            data_dir: Directory containing CSV files
            verbose: Whether to print detailed processing information
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.csv_files: List[Path] = []
        self.merged_df: Optional[pd.DataFrame] = None
        self.processing_log: List[str] = []
    
    def _log(self, message: str) -> None:
        """Log a message and optionally print it."""
        self.processing_log.append(message)
        if self.verbose:
            print(message)
    
    def discover_files(self) -> List[Path]:
        """
        Discover all CSV files in the data directory.
        
        Returns:
            List of Path objects for CSV files found
        """
        self.csv_files = list(self.data_dir.glob("*.csv"))
        
        if not self.csv_files:
            self._log(f"No CSV files found in {self.data_dir}")
            return []
        
        self._log(f"Found {len(self.csv_files)} CSV files:")
        for file in self.csv_files:
            self._log(f"  - {file.name}")
        
        return self.csv_files
    
    def _detect_and_process_date_column(self, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """
        Detect and process date columns in the DataFrame.
        
        Args:
            df: Input DataFrame
            file_name: Name of the source file for logging
            
        Returns:
            DataFrame with date column processed and set as index
        """
        # Handle unnamed first column that might contain dates
        if df.columns[0] == 'Unnamed: 0' or df.columns[0].startswith('Unnamed'):
            df = df.rename(columns={df.columns[0]: 'Date'})
            self._log(f"[{file_name}] Renamed first column to 'Date'")
        
        # Find date column (common names)
        date_col = None
        date_candidates = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'time', 'Time']
        
        for col in date_candidates:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is not None:
            # Found date column - use it as index
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                self._log(f"[{file_name}] Using '{date_col}' column as date")
                self._log(f"[{file_name}] Date range: {df.index.min()} to {df.index.max()}")
            except Exception as e:
                self._log(f"[{file_name}] Warning: Could not convert '{date_col}' to datetime: {e}")
                df.index.name = 'date'
        else:
            # No date column found - just name the index 'date'
            df.index.name = 'date'
            self._log(f"[{file_name}] No date column found, named index as 'date'")
        
        return df
    
    def _analyze_missing_values(self, df: pd.DataFrame, file_name: str) -> None:
        """
        Analyze and report missing values in the DataFrame.
        
        Args:
            df: DataFrame to analyze
            file_name: Name of the source file for logging
        """
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        if total_cells > 0:
            pct_missing = missing_cells / total_cells * 100
            self._log(f"[{file_name}] Missing values: {missing_cells:,} out of {total_cells:,} ({pct_missing:.1f}%)")
            
            if missing_cells > 0:
                self._log(f"[{file_name}] Missing values by column:")
                missing_by_col = df.isnull().sum()
                for col, count in missing_by_col[missing_by_col > 0].items():
                    pct = count / len(df) * 100
                    self._log(f"[{file_name}]   {col}: {count:,} ({pct:.1f}%)")
    
    def _process_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        Process a single CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Processed DataFrame
        """
        file_name = file_path.stem
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        self._log(f"\n--- Processing {file_path.name} ---")
        self._log(f"[{file_name}] Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Process date column
        df = self._detect_and_process_date_column(df, file_name)
        
        # Analyze missing values
        self._analyze_missing_values(df, file_name)
        
        # Add filename prefix to columns
        df.columns = [f"{file_name}_{col}" for col in df.columns]
        
        return df
    
    def merge_files(self, join_type: str = 'outer') -> Optional[pd.DataFrame]:
        """
        Merge all discovered CSV files.
        
        Args:
            join_type: Type of join to use ('outer', 'inner', 'left', 'right')
            
        Returns:
            Merged DataFrame or None if no files found
        """
        if not self.csv_files:
            self.discover_files()
        
        if not self.csv_files:
            return None
        
        self.merged_df = None
        
        for file_path in self.csv_files:
            df = self._process_single_file(file_path)
            
            # Merge with existing data
            if self.merged_df is None:
                self.merged_df = df
                self._log(f"[{file_path.stem}] Initial dataset: {len(df)} rows")
            else:
                rows_before = len(self.merged_df)
                self.merged_df = self.merged_df.join(df, how=join_type)
                rows_after = len(self.merged_df)
                self._log(f"[{file_path.stem}] After merge: {rows_before} -> {rows_after} rows")
            
            self._log(f"[{file_path.stem}] Added to merged dataset")
        
        if self.merged_df is not None:
            self._log(f"\nFinal merged dataset: {len(self.merged_df)} rows, {len(self.merged_df.columns)} columns")
            
            # Try to show date range
            try:
                self._log(f"Date range: {self.merged_df.index.min()} to {self.merged_df.index.max()}")
            except:
                self._log("Index contains mixed data types - cannot show date range")
        
        return self.merged_df
    
    def get_summary(self) -> dict:
        """
        Get a summary of the merged dataset.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.merged_df is None:
            return {"error": "No merged data available. Run merge_files() first."}
        
        summary = {
            "total_rows": len(self.merged_df),
            "total_columns": len(self.merged_df.columns),
            "files_processed": len(self.csv_files),
            "missing_values": self.merged_df.isnull().sum().sum(),
            "missing_percentage": (self.merged_df.isnull().sum().sum() / self.merged_df.size * 100) if self.merged_df.size > 0 else 0,
            "columns": self.merged_df.columns.tolist()
        }
        
        try:
            summary["date_range"] = {
                "start": self.merged_df.index.min(),
                "end": self.merged_df.index.max()
            }
        except:
            summary["date_range"] = "Mixed index types - cannot determine range"
        
        return summary
    
    def save_merged_data(self, output_path: str = "merged_data.csv") -> bool:
        """
        Save the merged DataFrame to a CSV file.
        
        Args:
            output_path: Path where to save the merged data
            
        Returns:
            True if successful, False otherwise
        """
        if self.merged_df is None:
            self._log("No merged data to save. Run merge_files() first.")
            return False
        
        try:
            self.merged_df.to_csv(output_path)
            self._log(f"Saved merged data to {output_path}")
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
    
    def clear_log(self) -> None:
        """Clear the processing log."""
        self.processing_log.clear()


# Usage example
if __name__ == "__main__":
    # Create merger instance
    merger = CSVMerger(data_dir="C:\\Users\\Maza\\Desktop\\USD-strenght-analysis\\data", verbose=True)
    
    # Discover and merge files
    merged_data = merger.merge_files(join_type='outer')
    
    if merged_data is not None:
        # Show summary
        summary = merger.get_summary()
        print(f"\nSummary: {summary}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(merged_data.head())
        
        # Save the merged data
        merger.save_merged_data("merged_data.csv")
    else:
        print("No data was merged.")