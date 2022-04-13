import pandas as pd
from datetime import timedelta

PATH = '../data/transactions_train.csv'
train = pd.read_csv(PATH)
train['t_dat'] = pd.to_datetime(train['t_dat'], format='%Y-%m-%d')
train['article_id'] = train['article_id'].astype('str')

gt_start = train['t_dat'].min() + timedelta(days=365)
gt_end = gt_start + timedelta(days=7)

gt_mask = (train['t_dat'] > gt_start) & (train['t_dat'] <= gt_end)
train = train[gt_mask]

gt = train.groupby(['customer_id'])['article_id'].apply(lambda x: ' '.join(list(x))).reset_index()

gt.to_csv('./predictions/gt_1yr_1wk.csv', index=False)