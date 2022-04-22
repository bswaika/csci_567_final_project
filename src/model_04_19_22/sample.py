import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from load import load_items, load_users, load_transactions, load_holdout
from constants import SAMPLE_DIR, SAMPLE_FILE

def generate_sample(frac=0.1, seed=False):
    users = load_users()
    if seed:
        user_sample = users.sample(frac=frac, random_state=9196).reset_index().drop(columns=['index'])
    else:
        user_sample = users.sample(frac=frac).reset_index().drop(columns=['index'])
    del users

    transactions = load_transactions()
    user_sample_mask = transactions['customer_id'].isin(user_sample['customer_id'])
    transaction_sample = transactions[user_sample_mask].reset_index().drop(columns=['index'])
    del transactions

    items = load_items()
    item_sample_ids = transaction_sample['article_id'].unique()
    item_sample_mask = items['article_id'].isin(item_sample_ids)
    item_sample = items[item_sample_mask]
    del items

    return item_sample, user_sample, transaction_sample

def generate_holdout_sample():
    holdout = load_holdout(4)
    items = load_items()
    users = load_users()

    item_sample = items[items['article_id'].isin(holdout['article_id'].unique())]
    user_sample = users[users['customer_id'].isin(holdout['customer_id'].unique())]
    
    return item_sample, user_sample, holdout

def generate_strategy_BNM(items: pd.DataFrame, users: pd.DataFrame, transactions: pd.DataFrame):
    '''
        Strategy - Buy/MultiBuy/NoBuy Signal for relevance (BNM)
        Relevance Score : 0 - Never interacted, 1 - Have interacted in the past or future but not in the next week, 2 - Have interacted in the next week with purchase_count 1 or 2, 3 - Have interacted in the next week with purchase count 3 or more
        Observation Date : Always 7 days before the transaction date
        User Features
        Item Features
        Sales Channel ID : 0 - If Relevance Score is 0 or 1, 1 or 2 - Depending on the sales channel value if Relevance Score is 2
    '''
    STRATEGY_STRING = 'BNM'
    sample_data = pd.DataFrame()

    user_item_cumulative_interactions = transactions.groupby('customer_id')['article_id'].apply(lambda x: set(x))
    transactions_by_user_and_time = transactions.groupby(['customer_id', 'timestamp'])

    for (user_id, timestamp), data in tqdm(transactions_by_user_and_time):
        user_items = user_item_cumulative_interactions.loc[user_id]
        current_items = set(data['article_id'].to_list())
        total_to_generate = len(current_items)

        relevance_1_item_ids = np.random.choice(list(user_items - current_items), min(total_to_generate // 2, len(user_items - current_items))).tolist()
        remaining = total_to_generate - len(relevance_1_item_ids)

        random_items = items.sample(max(100, total_to_generate * 4)).reset_index().drop(columns=['index'])
        relevance_mask_0 = ~random_items['article_id'].isin(list(user_items))
        relevance_0_item_ids = random_items[relevance_mask_0].sample(remaining).reset_index().drop(columns=['index'])['article_id'].to_list()
        
        interaction_counts = data.groupby('article_id')['timestamp'].count().reset_index()
        relevance_2_item_ids = interaction_counts.loc[interaction_counts['timestamp'] <= 2, 'article_id'].to_list()
        relevance_3_item_ids = interaction_counts.loc[interaction_counts['timestamp'] > 2, 'article_id'].to_list()
        
        relevance_3_interactions = data[data['article_id'].isin(relevance_3_item_ids)].reset_index().drop(columns=['index'])
        relevance_3_interactions['relevance'] = 3
        relevance_3_interactions['relevance'] = relevance_3_interactions['relevance'].astype('uint8')
        relevance_3_interactions = relevance_3_interactions.drop_duplicates(ignore_index=True)
        
        relevance_2_interactions = data[data['article_id'].isin(relevance_2_item_ids)].reset_index().drop(columns=['index'])
        relevance_2_interactions['relevance'] = 2
        relevance_2_interactions['relevance'] = relevance_2_interactions['relevance'].astype('uint8')
        relevance_2_interactions = relevance_2_interactions.drop_duplicates(ignore_index=True)
        
        relevance_1_interactions = pd.DataFrame({
            'customer_id': pd.Series([user_id] * len(relevance_1_item_ids), dtype='str'),
            'timestamp': pd.Series([timestamp] * len(relevance_1_item_ids), dtype='uint32'),
            'sales_channel_id': pd.Series([0] * len(relevance_1_item_ids), dtype='int8'),
            'article_id': pd.Series(relevance_1_item_ids, dtype='str'),
            'relevance': pd.Series([1] * len(relevance_1_item_ids), dtype='uint8')
        })
        
        relevance_0_interactions = pd.DataFrame({
            'customer_id': pd.Series([user_id] * len(relevance_0_item_ids), dtype='str'),
            'timestamp': pd.Series([timestamp] * len(relevance_0_item_ids), dtype='uint32'),
            'sales_channel_id': pd.Series([0] * len(relevance_0_item_ids), dtype='int8'),
            'article_id': pd.Series(relevance_0_item_ids, dtype='str'),
            'relevance': pd.Series([0] * len(relevance_0_item_ids), dtype='uint8')
        })
        
        user_interactions = pd.DataFrame().append(relevance_3_interactions).append(relevance_2_interactions).append(relevance_1_interactions).append(relevance_0_interactions).reset_index().drop(columns=['index'])
        user_interactions = user_interactions.sample(frac=1).reset_index().drop(columns=['index'])
        sample_data = sample_data.append(user_interactions)
    
    sample_data = sample_data.reset_index().drop(columns=['index'])
    sample_data['timestamp'] = sample_data['timestamp'] - (60 * 60 * 24 * 7)
    
    sample_data = sample_data.merge(items, on='article_id')
    sample_data = sample_data.merge(users, on='customer_id')    

    file_id = np.random.randint(10e5, 10e6) if not HOLDOUT else 'holdout'
    files = os.listdir(SAMPLE_DIR)

    filename = SAMPLE_FILE.format(dir=SAMPLE_DIR, id=file_id, strategy=STRATEGY_STRING)
    while filename in files:
        file_id += 1
        filename = SAMPLE_FILE.format(dir=SAMPLE_DIR, id=file_id, strategy=STRATEGY_STRING)
    
    sample_data.to_csv(filename, index=False)
    sample_data.info(verbose=True, memory_usage='deep')
    print(f'wrote {sample_data.shape[0]} rows of synthetic data...')

HOLDOUT = False
ROUNDS_TO_GENERATE = 5
SAMPLING_FRACTION = 0.015

for i in range(1 if HOLDOUT else ROUNDS_TO_GENERATE):
    print(f'started round {i+1}...')
    # Generate a sample of items, users, transactions based on the fraction
    items, users, transactions = generate_holdout_sample() if HOLDOUT else generate_sample(SAMPLING_FRACTION)
    print(f'generated sample with {items.shape[0]} items, {users.shape[0]} users, and {transactions.shape[0]} transactions...')

    # Drop features currently not in use
    transactions = transactions.drop(columns=['t_dat'])
    items = items.drop(columns=['prod_name', 'detail_desc'])

    # Extract average price of an item in the sample
    item_price = transactions.groupby('article_id')['price'].mean().reset_index().rename(columns={'price': 'avg_price'})
    items = items.merge(item_price, on='article_id')
    transactions = transactions.drop(columns='price')

    # Extract popularity of an item in the sample
    item_popularity = transactions.groupby('article_id')['timestamp'].count().reset_index().rename(columns={'timestamp': 'popularity'})
    total_purchases = item_popularity['popularity'].sum()
    item_popularity['popularity'] = item_popularity['popularity'] / total_purchases
    items = items.merge(item_popularity, on='article_id')

    # Generate data for ranking system based on current sample and specific strategy
    generate_strategy_BNM(items, users, transactions)
    print(f'completed round {i+1}...')