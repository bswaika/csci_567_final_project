from time import time
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

items = items.drop(columns=['prod_name', 'detail_desc'])

prediction_date = pd.Series(['2020-09-22'])
prediction_date = pd.to_datetime(prediction_date, format=DATETIME_FORMAT)
prediction_date = prediction_date.view('int64') // 10 ** 9
prediction_date = prediction_date.values[0]
prediction_date = (prediction_date - scalers['timestamp'][0]) / scalers['timestamp'][1]
users['timestamp'] = prediction_date

# for user in users.values[:1]:
#     item_sample = items.sample(frac=0.002).reset_index().drop(columns=['index'])
#     user = pd.DataFrame(np.repeat(user.reshape(-1, 1), item_sample.shape[0], axis=1).T, columns=users.columns)
#     data_point = pd.concat([user, item_sample], axis=1, ignore_index=True)
#     data_point.columns = list(user.columns) + list(item_sample.columns)
#     print(data_point.dtypes)

popular_items = pd.read_csv('./data/top_7000_articles.csv')
user_queries = pd.read_csv('./data/user_queries.csv')

user_queries['unique_articles'] = user_queries['unique_articles'].apply(lambda x: str(x).replace('[', '').replace(']', '').replace("'", '').split(', '))
lengths = user_queries['unique_articles'].apply(lambda x: len(x))
max_length, min_length = lengths.max(), lengths.min()
min_ratio, max_ratio = 0.1, 0.75

# test = users.loc[0:4].merge(items.sample(10).reset_index().drop(columns=['index']), how='cross')

BATCH_SIZE = 1000
ITEM_SAMPLE_SIZE = 12_000


def get_products_of_interest(interest, col, items_to_rank, num_products_per_interest):
    if num_products_per_interest <= 0:
        return set()
    mask = (items[col] == interest) & ~(items['article_id'].isin(items_to_rank))
    interest_sample = items[mask].sample(min(items[mask].shape[0], num_products_per_interest)).reset_index()
    return set(interest_sample['article_id'].to_list())

def sampler(row):
    already_bought = row['unique_articles']
    items_to_rank = set(already_bought)
    remaining = ITEM_SAMPLE_SIZE - len(already_bought)
    if remaining <= 0:
        return list(items_to_rank)
    prod_type_name_query = row['query'].split('|')[2]
    col, interests = prod_type_name_query.split(':')
    interests = interests.split(',')
    num_interest_matching_products = int(((((len(already_bought) - min_length) / (max_length - min_length)) * (max_ratio - min_ratio)) + min_ratio) * remaining)
    num_products_per_interest = num_interest_matching_products // len(interests)
    for subset in map(lambda i: get_products_of_interest(i, col, items_to_rank, num_products_per_interest), interests):
        items_to_rank |= subset
        remaining -= len(subset)
    if remaining <= 0:
        return list(items_to_rank)
    items_to_rank |= set(popular_items[~popular_items['article_id'].isin(items_to_rank)].sample(min(3500, remaining)).reset_index()['article_id'].to_list())
    remaining -= min(3500, remaining)
    if remaining > 0:
        items_to_rank |= set(items[~items['article_id'].isin(items_to_rank)].sample(remaining).reset_index()['article_id'].to_list())
    return list(map(str, items_to_rank))

predictions = pd.DataFrame()
with tqdm(total=users.shape[0]) as progress:
    for i in range(0, users.shape[0], BATCH_SIZE):
        user_batch = users.loc[i:i+BATCH_SIZE-1].reset_index().drop(columns=['index'])
        user_batch_queries = user_queries[user_queries['customer_id'].isin(user_batch['customer_id'])]
        user_batch_queries['items'] = user_batch_queries.apply(sampler, axis=1)
        for j in range(user_batch.shape[0]):
            user_id = user_batch.loc[j, 'customer_id']
            item_ids = user_batch_queries[user_batch_queries['customer_id'] == user_id].reset_index()['items'].values[0]
            user_items = items[items['article_id'].isin(item_ids)].reset_index().drop(columns=['index'])
            user_data = user_batch[user_batch['customer_id'] == user_id].reset_index().drop(columns=['index']).merge(user_items, how='cross')
            user_data['customer_id'] = encoder.transform(user_data['customer_id'].to_numpy().reshape(-1, 1)).flatten()
            user_data['article_id'] = user_data['article_id'].astype('int64')
            user_pred = booster.predict(user_data)
            item_pred = list(sorted(zip(item_ids, user_pred), key=lambda x: x[1], reverse=True))[:30]
            item_pred = list(map(lambda x: (f'0{x[0]}', x[1]), item_pred))
            pred = pd.DataFrame({
                'customer_id': encoder.inverse_transform(np.array([user_data['customer_id'].values[0]] * 30).reshape(-1, 1)).flatten(),
                'predictions': item_pred
            })
            pred['article_id'] = pred['predictions'].apply(lambda x: x[0])
            pred['rank'] = pred['predictions'].apply(lambda x: x[1])
            pred['rank'] = pred['rank'].astype('float32')
            pred = pred.drop(columns='predictions')
            
            predictions = predictions.append(pred)
            progress.update(1)

ts = int(time())
predictions.reset_index().drop(columns=['index']).to_csv(f'./submissions/master_{ts}_BNM.csv', index=False)