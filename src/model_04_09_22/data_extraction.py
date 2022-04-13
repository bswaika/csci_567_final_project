'''
    Script
    ----------------------------------
    Does the following:
    1. Extract TRAIN_DATA_DURATION_DAYS of transactions data
    2. Write out to disk
'''

from constants import RAW_DATA_FILES, DATA_FILES, DATA_DIR
from common import load_df, save_df, is_files_exist_in
from debug import is_debug_mode, get_debug_config, is_forced

import transactions

if __name__ == '__main__':
    DEBUG = is_debug_mode()
    
    # Perform operations only when forced or data files don't exist
    if is_forced() or not is_files_exist_in(DATA_DIR, ['transactions']):
        
        # Get debug config based on whether it is turned on or off
        config = get_debug_config(DEBUG)
        
        # Load data frame
        transactions_df = load_df(RAW_DATA_FILES['transactions'], config['nrows'])
        
        # Extract subset of transactions with defaults
        transactions_df = transactions.extract_subset(transactions.set_date_column_type(transactions_df))
        
        # Write subset to disk
        save_df(transactions_df, DATA_FILES['transactions'])
        

