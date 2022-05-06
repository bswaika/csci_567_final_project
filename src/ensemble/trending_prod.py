import pandas as pd
import numpy as np
from math import sqrt
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

N = 12
df_trans = pd.read_csv('../../data/transactions_train.csv',dtype={'article_id': str})
df_trans['t_dat'] = pd.to_datetime(df_trans['t_dat'])

df = df_trans[['t_dat', 'customer_id', 'article_id']].copy()
last_ts = df['t_dat'].max()
df['ldbw'] = df['t_dat'].apply(lambda d: last_ts - (last_ts - d).floor('7D'))
weekly_sales = df.drop('customer_id', axis=1).groupby(['ldbw', 'article_id']).count()
weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})
df = df.join(weekly_sales, on=['ldbw', 'article_id'])
weekly_sales = weekly_sales.reset_index().set_index('article_id')
last_day = last_ts.strftime('%Y-%m-%d')

df = df.join(
    weekly_sales.loc[weekly_sales['ldbw']==last_day, ['count']],
    on='article_id', rsuffix="_targ")

df['count_targ'].fillna(0, inplace=True)
del weekly_sales
df['quotient'] = df['count_targ'] / df['count']

purchase_dict = {}

for i in tqdm(df.index):
    cust_id = df.at[i, 'customer_id']
    art_id = df.at[i, 'article_id']
    t_dat = df.at[i, 't_dat']

    if cust_id not in purchase_dict:
        purchase_dict[cust_id] = {}

    if art_id not in purchase_dict[cust_id]:
        purchase_dict[cust_id][art_id] = 0
    
    x = max(1, (last_ts - t_dat).days)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    y = a / np.sqrt(x) + b * np.exp(-c*x) - d

    value = df.at[i, 'quotient'] * max(0, y)
    purchase_dict[cust_id][art_id] += value

target_sales = df.drop('customer_id', axis=1).groupby('article_id')['quotient'].sum()
general_pred = target_sales.nlargest(N).index.tolist()

pairs = np.load('./pairs_cudf.npy',allow_pickle=True).item()
sub = pd.read_csv('../../data/sample_submission.csv')

pred_list = []
for cust_id in tqdm(sub['customer_id']):
    if cust_id in purchase_dict:
        series = pd.Series(purchase_dict[cust_id])
        series = series[series > 150]
        l = series.nlargest(N).index.tolist()
        tmp_l = l.copy()
        for elm in tmp_l:
            if len(l) < N and int(elm) in pairs.keys():
                itm = pairs[int(elm)]
                l.append('0' + str(itm))
        if len(l) < N:
            l = l + general_pred[:(N-len(l))]
    else:
        l = general_pred
    pred_list.append(' '.join(l))

sub['prediction'] = pred_list
sub.to_csv('./submissions/trending_prod.csv',index=False)