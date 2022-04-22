import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from load import load_users

users = load_users()
encoder = OrdinalEncoder().fit(users['customer_id'].to_numpy().reshape(-1, 1))

DIR = './submissions/master_04_21_22'
files = os.listdir(DIR)

predictions = pd.DataFrame()
for file in tqdm(files):
    df = pd.read_csv(f'{DIR}/{file}')
    df['customer_id'] = encoder.inverse_transform(df['customer_id'].to_numpy().reshape(-1, 1)).flatten()
    df = df.sort_values(by=['customer_id', 'rank'], ascending=False)
    predictions = predictions.append(
        df.groupby('customer_id')['article_id']
        .agg(lambda x: ' '.join(map(lambda i: f'0{i}', list(x)[:12])))
        .reset_index()
    )
    df.to_csv(f'{DIR}/{file}', index=False)

SUBMISSION_DIR = './submissions'
predictions.reset_index().drop(columns=['index']).to_csv(f'{SUBMISSION_DIR}/submission_merged_c25rcpop.csv', index=False)