import logging
import sys
from data_agrgregator import CSVMerger
from data_cleaning import BusinessDaysProcessor

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    import os
    if not os.path.exists('logs'):
        os.makedirs('logs')
    


    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized")
    return logger

def main():
    # Setup logging
    logger = setup_logging()
    
    try:
        logger.info("Starting data processing pipeline")
        
        # Path to the directory with CSV files
        data_directory = "C:\\Users\\Maza\\Desktop\\USD-strenght-analysis\\data"
        logger.info(f"Data directory: {data_directory}")

        # Initialize merger
        logger.info("Initializing CSVMerger")
        merger = CSVMerger(data_dir=data_directory, verbose=True)

        # Merge files
        logger.info("Starting file merge process")
        merged_df = merger.merge_files(join_type='outer')

        if merged_df is not None:
            logger.info(f"Successfully merged files. DataFrame shape: {merged_df.shape}")
            
            # Print summary
            summary = merger.get_summary()
            logger.info("Merger summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")

            # Print first few rows
            logger.info("First 5 rows of merged data:")
            print(merged_df.head())

            # Save to CSV
            logger.info("Attempting to save merged data")
            if merger.save_merged_data("merged_data.csv"):
                logger.info("Merged data saved successfully to 'merged_data.csv'")
            else:
                logger.error("Failed to save merged data")
                
        else:
            logger.warning("No CSV files were found or merged")
            return
        
        # Business days processing
        logger.info("Starting business days processing")
        processor = BusinessDaysProcessor(verbose=True)
        
        logger.info("Preparing business days data")
        df_clean = processor.prepare_business_days_data(merged_df)
        missing_dates = processor.find_missing_dates("commodities_Agriculture ETF")
        print(missing_dates)
        
        if df_clean is not None:
            logger.info(f"Business days processing completed. Clean data shape: {df_clean.shape}")
            
            # Analyze the results
            logger.info("Data info for cleaned dataset:")
            print(df_clean.info())

            # Save the processed data
            logger.info("Saving processed data")
            if processor.save_processed_data("clean_data.csv"):
                logger.info("Processed data saved successfully to 'clean_data.csv'")
            else:
                logger.error("Failed to save processed data")
        else:
            logger.error("Business days processing failed - returned None")
            
        logger.info("Data processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    
    finally:
        logger.info("Main execution finished")

if __name__ == "__main__":
    main()