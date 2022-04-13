import pandas as pd
import numpy as np
from datetime import timedelta

def load_transactions(filepath, nrows=None):
    transactions = pd.read_csv(filepath, nrows=nrows)
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], format='%Y-%m-%d')
    return transactions

def train_test_split(transactions: pd.DataFrame, days=7):
    end_date = transactions['t_dat'].max() - timedelta(days=days)
    mask = transactions['t_dat'] <= end_date
    return transactions[mask], transactions[~mask]

def transform_transactions(trx: pd.DataFrame, selectors=['freq', 'spend_freq']):
    articles = pd.read_csv('./data/top_1000_articles.csv')
    customers = pd.read_csv('./data/top_300_000_customers.csv')

    trx = trx.groupby(['customer_id', 'article_id'])[['t_dat', 'price']].agg({'t_dat': 'count', 'price': 'mean'}).reset_index().rename(columns={'t_dat': 'qty', 'price': 'avg_price'})
    trx['spend'] = trx['qty'] * trx['avg_price']
    totals = trx.groupby('customer_id')[['qty', 'spend']].sum().reset_index().rename(columns={'qty': 'total_qty', 'spend': 'total_spend'})
    trx = trx.merge(totals, on='customer_id')
    trx['spend_freq'] = trx['spend'] / trx['total_spend']
    trx['freq'] = trx['qty'] / trx['total_qty']
    
    trx = trx[trx['customer_id'].isin(customers['customer_id']) & trx['article_id'].isin(articles['article_id'])]

    selectors = ['customer_id', 'article_id'] + selectors
    trx_features = trx[selectors].astype({'customer_id': np.object0, 'article_id': np.int64, 'freq': np.float32, 'spend_freq': np.float32})
    trx_features = [np.array(v).reshape(trx.shape[0], 1) for n, v in trx_features.items()]

    return trx_features

if __name__ == '__main__':
    transactions = load_transactions('./data/transactions_popular.csv')
    train, test = train_test_split(transactions)
    print(transform_transactions(test))
    print(transform_transactions(train))
    