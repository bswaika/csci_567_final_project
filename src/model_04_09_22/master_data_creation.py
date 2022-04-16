'''
    Script
    ----------------------------------
    Does the following:
    1. Read local transactions data
    2. Compute popular users and items
    3. Prepare users and items data
        a. Keywords for items
        b. Cleaning and wrangling users
    4. Aggregate transactions data
    5. Write out prepared master data for hypothesis testing
'''

from constants import NUM_ITEMS, NUM_USERS, DATA_FILES, RAW_DATA_FILES, DATA_DIR
from debug import is_debug_mode, get_debug_config, is_forced
from common import load_df, save_df, is_files_exist_in, select_columns

import transactions, items, users

if __name__ == '__main__':
    DEBUG = is_debug_mode()
    
    # Perform operations only when forced or data files don't exist
    if is_forced() or not is_files_exist_in(DATA_DIR, ['customers', 'articles', 'master_OR', 'master_AND']):
        
        # Get debug config based on whether it is turned on or off
        config = get_debug_config(DEBUG)
        
        # Load transactions data from local data directory
        transactions_df = load_df(DATA_FILES['transactions'], config['nrows'])
        transactions_df = select_columns(transactions_df, ['customer_id', 'article_id', 't_dat'])
        
        # Load items data from base data directory
        items_raw_df = load_df(RAW_DATA_FILES['items'])
        items_raw_df = select_columns(items_raw_df, ['article_id'] + [col for col in items_raw_df.columns if col != 'detail_desc' and not any([x in col for x in ['id', 'no', 'code']])])
        
        # Load users data from base data directory
        users_raw_df = load_df(RAW_DATA_FILES['users'])
        users_raw_df = select_columns(users_raw_df, ['customer_id', 'age', 'club_member_status', 'postal_code'])
        
        # Compute popular items and users
        popular_users = transactions.compute_most_popular(transactions_df, 'customer_id', NUM_USERS)
        popular_items = transactions.compute_most_popular(transactions_df, 'article_id', NUM_ITEMS)

        save_df(popular_users, DATA_FILES['pop_users'])
        save_df(popular_items, DATA_FILES['pop_items'])
        
        # Remove all transactions which don't include either a popular item or a popular user
        transactions_df = transactions.extract_subset_union(transactions_df, popular_users['customer_id'].to_numpy(), popular_items['article_id'].to_numpy())

        # Wrangle raw dataframes for users and items
        users_raw_df = users.fill(users_raw_df)
        items_raw_df = items.generate_keywords(items_raw_df)
        items_raw_df = select_columns(items_raw_df, ['article_id', 'prod_name', 'keywords'])

        # Prepare master dataframe by merging
        master_OR_df = transactions_df.merge(items_raw_df, on='article_id')
        master_OR_df = master_OR_df.merge(users_raw_df, on='customer_id')

        save_df(master_OR_df, DATA_FILES['master_union'])

        # Remove all items and users from their respective dataframes if they aren't popular
        users_df = users_raw_df[users_raw_df['customer_id'].isin(popular_users['customer_id'].to_numpy())]
        items_df = items_raw_df[items_raw_df['article_id'].isin(popular_items['article_id'].to_numpy())]

        # Free up space
        del users_raw_df
        del items_raw_df

        # Write users and items to disk
        save_df(users_df, DATA_FILES['users'])
        save_df(items_df, DATA_FILES['items'])

        # Aggregate transactions data
        # transactions_df = transactions.compute_targets(transactions_df)
        # transactions_df = select_columns(transactions_df, ['customer_id', 'article_id', 'freq', 'spend_freq'])
        transactions_df = transactions.extract_subset_intersection(transactions_df, popular_users['customer_id'].to_numpy(), popular_items['article_id'].to_numpy())
        
        # Prepare master dataframe by merging
        master_AND_df = transactions_df.merge(items_df, on='article_id')
        master_AND_df = master_AND_df.merge(users_df, on='customer_id')

        # Write master dataframe to disk
        save_df(master_AND_df, DATA_FILES['master_intersection'])



