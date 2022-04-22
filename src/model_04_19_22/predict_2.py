import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm import Booster
from sklearn.preprocessing import OrdinalEncoder
from constants import MODEL_FILE, MODEL_DIR, DATETIME_FORMAT
from load import load_users, load_items

booster = Booster(model_file=MODEL_FILE.format(dir=MODEL_DIR, timestamp='1650539122', strategy='BNM'))
users = load_users()
items = load_items()
# transactions = load_transactions()

numerical_cols = ['FN', 'Active', 'sales_channel_id']
categories = {}
with open('./data/train_categories.txt', 'r') as infile:
    lines = infile.readlines()
    for line in lines:
        line = line.strip()
        col, category_list = line.split(':')
        if col in numerical_cols:
            categories[col] = pd.CategoricalDtype(categories=list(map(int, category_list.split(','))))
        else:
            categories[col] = pd.CategoricalDtype(categories=category_list.split(','))

scalers = {}
with open('./data/train_scalers.txt', 'r') as infile:
    lines = infile.readlines()
    for line in lines:
        col, stats = line.split(':')
        scalers[col] = list(map(float, stats.split(',')))

for col in users.columns:
    if col in categories:
        users[col] = pd.Series(pd.Categorical(users[col], dtype=categories[col]))

for col in items.columns:
    if col in categories:
        items[col] = pd.Series(pd.Categorical(items[col], dtype=categories[col]))

items_features = pd.read_csv('./data/item_features_new.csv')
items_features['article_id'] = items_features['article_id'].astype('str')
items = items.merge(items_features, on='article_id')

for key in scalers:
    if key in items.columns:
        items[key] = (items[key] - scalers[key][0]) / scalers[key][1]
    if key in users.columns:
        users[key] = (users[key] - scalers[key][0]) / scalers[key][1]

encoder = OrdinalEncoder().fit(users['customer_id'].to_numpy().reshape(-1, 1))
# users['customer_id'] = encoder.transform(users['customer_id'].to_numpy().reshape(-1, 1)).flatten()
users['customer_id'] = encoder.transform(users['customer_id'].to_numpy().reshape(-1, 1)).flatten()


items = items.drop(columns=['prod_name', 'detail_desc'])

prediction_date = pd.Series(['2020-09-22'])
prediction_date = pd.to_datetime(prediction_date, format=DATETIME_FORMAT)
prediction_date = prediction_date.view('int64') // 10 ** 9
prediction_date = prediction_date.values[0]
prediction_date = (prediction_date - scalers['timestamp'][0]) / scalers['timestamp'][1]
users['timestamp'] = prediction_date

start, end = int(sys.argv[2]), int(sys.argv[3])

users = users.loc[start:end].reset_index().drop(columns=['index'])

write_interval, j, counter = 50_000, 0, 0
predictions = pd.DataFrame()
for i in tqdm(range(users.shape[0])):
    user_id = users.loc[i, 'customer_id']
    item_sample = items.sample(500).reset_index().drop(columns=['index'])
    user = users[users['customer_id'] == user_id].reset_index().drop(columns=['index']).merge(item_sample, how='cross')
    user['article_id'] = user['article_id'].astype('int64')
    user_pred = booster.predict(user)
    pred = pd.DataFrame({
        'customer_id': encoder.inverse_transform(user['customer_id'].to_numpy().reshape(-1, 1)).flatten(),
        'article_id': user['article_id'].to_numpy(),
        'rank': user_pred
    })
    predictions.append(pred)
    if j < write_interval:
        j += 1
    else:
        predictions.reset_index().drop(columns=['index']).to_csv(f'./submissions/master_04_21_22/master_{counter}_{sys.argv[1]}.csv', index=False)
        predictions = pd.DataFrame()
        j = 0
        counter += 1

if 0 < j <= write_interval:
    predictions.reset_index().drop(columns=['index']).to_csv(f'./submissions/master_04_21_22/master_{counter}_{sys.argv[1]}.csv', index=False)