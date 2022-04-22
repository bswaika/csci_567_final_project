import os
from datetime import timedelta
import pandas as pd
from constants import RAW_DATA_FILES, DATETIME_FORMAT, DATA_FILES, DATA_DIR

def load_items() -> pd.DataFrame:
    items = pd.read_csv(RAW_DATA_FILES['items'])
    items['article_id'] = items['article_id'].astype('str')
    drop_columns = [col for col in items.columns if col != 'article_id' and any(['no' in col, 'code' in col, 'id' in col])]
    items = items.drop(columns=drop_columns)

    item_cat_cols = [col for col in items.columns if col not in ['article_id', 'prod_name', 'detail_desc']]
    for col in item_cat_cols:
        items[col] = items[col].astype('category')
    
    print(f'loaded {items.shape[0]} items with {items.shape[1]} columns...')    
    return items

def load_users() -> pd.DataFrame:
    users = pd.read_csv(RAW_DATA_FILES['users'])
    users['age'] = users['age'].fillna(-1).astype('int8')
    users['FN'] = users['FN'].fillna(0).astype('int8')
    users['Active'] = users['Active'].fillna(0).astype('int8')
    users['club_member_status'] = users['club_member_status'].fillna('NONE')
    users['fashion_news_frequency'] = users['fashion_news_frequency'].fillna('NONE')

    user_cat_cols = ['FN', 'Active', 'club_member_status', 'fashion_news_frequency']
    for col in user_cat_cols:
        users[col] = users[col].astype('category')

    print(f'loaded {users.shape[0]} users with {users.shape[1]} columns...')
    return users

def load_transactions_file(filepath) -> pd.DataFrame:
    transactions = pd.read_csv(filepath)
    transactions['sales_channel_id'] = transactions['sales_channel_id'].astype('int8')
    transactions['price'] = transactions['price'].astype('float32')
    transactions['article_id'] = transactions['article_id'].astype('str')
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], format=DATETIME_FORMAT)
    transactions['timestamp'] = transactions['t_dat'].view('int64') // 10 ** 9
    transactions['timestamp'] = transactions['timestamp'].astype('uint32')
    transactions = transactions.sort_values(by='t_dat')

    print(f'loaded {transactions.shape[0]} transactions with {transactions.shape[1]} columns...')

    return transactions

def load_transactions(holdout=True, holdout_dur_weeks=4) -> pd.DataFrame:    
    return load_transactions_holdout(holdout, holdout_dur_weeks)

def load_holdout(holdout_dur_weeks) -> pd.DataFrame:
    return load_transactions_file(DATA_FILES['holdout'].format(dir=DATA_DIR, weeks=holdout_dur_weeks))

def load_transactions_holdout(mask_holdout, holdout_dur_weeks) -> pd.DataFrame:
    transactions = load_transactions_file(RAW_DATA_FILES['transactions'])

    if not mask_holdout:
        return transactions
    else:
        end_date = transactions['t_dat'].max()
        holdout_start_date = end_date - timedelta(days=holdout_dur_weeks*7)
        holdout_mask = transactions['t_dat'] >= holdout_start_date

        if not is_holdout_exists(holdout_dur_weeks):
            transactions[holdout_mask].to_csv(DATA_FILES['holdout'].format(dir=DATA_DIR, weeks=holdout_dur_weeks), index=False, columns=[col for col in transactions.columns if col != 'timestamp'])
            print('created holdout file...')

        return transactions[~holdout_mask]

def is_holdout_exists(holdout_dur_weeks):
    file = DATA_FILES['holdout'].format(dir=DATA_DIR, weeks=holdout_dur_weeks).split('/')[-1]
    dir_listing = os.listdir(DATA_DIR)
    return file in dir_listing