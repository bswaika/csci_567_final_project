# import pandas as pd
# from constants import RAW_DATA_FILES

# transactions = pd.read_csv(RAW_DATA_FILES['transactions'])
# transactions.t_dat = pd.to_datetime(transactions.t_dat)
# transactions["week"] = 104 - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7

# USE_WEEKS = 10
# TEST_WEEK = 104

# valid = transactions[transactions['week'] == TEST_WEEK][['customer_id', 'article_id']]
# transactions = transactions[(transactions.week > TEST_WEEK - USE_WEEKS) & (transactions.week < TEST_WEEK)] 

# previous_week = transactions[transactions['week'] == 103][['customer_id', 'article_id']]
# top_products = pd.DataFrame(data=transactions[transactions['week'] == 103].value_counts('article_id').iloc[:12].index.tolist(),
#                             columns=['article_id'])
# previous_week_top = transactions[['customer_id']].drop_duplicates().merge(top_products, how='cross')
# cand = pd.concat([previous_week, previous_week_top]).drop_duplicates().reset_index().drop(columns=['index'])

# print(cand.groupby('customer_id')['article_id'].count())

# import pandas as pd
# import numpy as np
# import os
# from tqdm import tqdm
# from reco.recommender import FunkSVD
# from reco.metrics import rmse
# import datetime
# from collections import Counter
# from datetime import timedelta

# train = pd.read_csv('../../data/transactions_train.csv', dtype={'article_id':str})

# uid = 'customer_id'
# iid = 'article_id'
# time_col = 't_dat'

# train[time_col] = pd.to_datetime(train[time_col])

# one_day = timedelta(days=1)
# year_month_day = str(train[time_col].max() - one_day*7*8)[:10].split('-')

# year_ = int(year_month_day[0])
# month_ = int(year_month_day[1])
# day_ = int(year_month_day[2])

# train_pop = train.loc[train[time_col] >= datetime.datetime(year_, month_, day_)]

# train_pop['diff'] = (train_pop[time_col].max() - train_pop[time_col])
# train_pop['pop_factor'] = 1 / (train_pop['diff'].dt.days + 1)

# popular_items_group = train_pop.groupby([iid])['pop_factor'].sum()
# _, popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])

# def get_most_freq_next_item(user_group):
#     next_items = {}
#     for user in tqdm(user_group.keys()):
#         items = user_group[user]
#         for i,item in enumerate(items[:-1]):
#             if item not in next_items:
#                 next_items[item] = []
# #             if item != items[i+1]:
# #                 next_items[item].append(items[i+1])
#             next_items[item].append(items[i+1])
    
#     pred_next = {}
#     for item in next_items:
#         if len(next_items[item]) >= 5:
#             most_common = Counter(next_items[item]).most_common()
#             ratio = most_common[0][1]/len(next_items[item])
#             if ratio >= 0.1:
#                 pred_next[item] = most_common[0][0]
            
#     return pred_next

# one_day = timedelta(days=1)
# year_month_day = str(train[time_col].max() - one_day*7*4)[:10].split('-')

# year_ = int(year_month_day[0])
# month_ = int(year_month_day[1])
# day_ = int(year_month_day[2])

# user_group = train.loc[train[time_col] >= datetime.datetime(year_, month_, day_)].groupby([uid])[iid].apply(list)
# pred_next = get_most_freq_next_item(user_group)
# user_group_dict = user_group.to_dict()

# one_day = timedelta(days=1)
# year_month_day = str(train[time_col].max() - one_day*7*16)[:10].split('-')

# year_ = int(year_month_day[0])
# month_ = int(year_month_day[1])
# day_ = int(year_month_day[2])

# df = train.loc[train[time_col] >= datetime.datetime(year_, month_, day_), [uid, iid, time_col]].copy()

# df['diff'] = (df[time_col].max() - df[time_col])
# df['pop_factor'] = 1 / (df['diff'].dt.days + 1)

# popular_items_group = df.groupby([iid])['pop_factor'].sum()

# # purchase count of each article
# items_total_count = df.groupby([iid])[iid].count()
# # purchase count of each user
# users_total_count = df.groupby([uid])[uid].count()

# df['feedback'] = 1
# df = df.groupby([uid, iid]).sum().reset_index()
# df['feedback'] = df.apply(lambda row: row['feedback']/popular_items_group[row[iid]], axis=1)

# df['feedback'] = df['feedback'].apply(lambda x: 5.0 if x>5.0 else x)

# df = df[[uid, iid, 'feedback']]

# # shuffling
# df = df.sample(frac=1).reset_index(drop=True)

# svd = FunkSVD(k=8, learning_rate=0.008, regularizer=0.01, iterations=80, method='stochastic', bias=True)
# svd.fit(X=df, formatizer={'user':uid, 'item':iid, 'value':'feedback'},verbose=True)

# one_day = timedelta(days=1)

# year_month_day_w1 = str(train[time_col].max() - one_day*7)[:10].split('-')
# year_w1 = int(year_month_day_w1[0])
# month_w1 = int(year_month_day_w1[1])
# day_w1 = int(year_month_day_w1[2])
# print(year_w1, month_w1, day_w1)


# year_month_day_w2 = str(train[time_col].max() - one_day*7*2)[:10].split('-')
# year_w2 = int(year_month_day_w2[0])
# month_w2 = int(year_month_day_w2[1])
# day_w2 = int(year_month_day_w2[2])
# print(year_w2, month_w2, day_w2)


# year_month_day_w3 = str(train[time_col].max() - one_day*7*3)[:10].split('-')
# year_w3 = int(year_month_day_w3[0])
# month_w3 = int(year_month_day_w3[1])
# day_w3 = int(year_month_day_w3[2])
# print(year_w3, month_w3, day_w3)


# year_month_day_w4 = str(train[time_col].max() - one_day*7*4)[:10].split('-')
# year_w4 = int(year_month_day_w4[0])
# month_w4 = int(year_month_day_w4[1])
# day_w4 = int(year_month_day_w4[2])
# print(year_w4, month_w4, day_w4)

# train1 = train.loc[(train[time_col] >= datetime.datetime(year_w1, month_w1, day_w1))]
# train2 = train.loc[(train[time_col] >= datetime.datetime(year_w2, month_w2, day_w2)) & (train[time_col] < datetime.datetime(year_w1, month_w1, day_w1))]
# train3 = train.loc[(train[time_col] >= datetime.datetime(year_w3, month_w3, day_w3)) & (train[time_col] < datetime.datetime(year_w2, month_w2, day_w2))]
# train4 = train.loc[(train[time_col] >= datetime.datetime(year_w4, month_w4, day_w4)) & (train[time_col] < datetime.datetime(year_w3, month_w3, day_w3))]

# tmp = train1.groupby([uid,iid])[time_col].agg('count').reset_index()
# tmp.columns = [uid,iid,'cnt']
# train1 = train1.merge(tmp, on = [uid,iid], how='left')
# train1 = train1.sort_values([time_col, 'cnt'],ascending=False)
# train1.index = range(len(train1))
# positive_items_per_user1 = train1.groupby([uid])[iid].apply(list)


# tmp = train2.groupby([uid,iid])[time_col].agg('count').reset_index()
# tmp.columns = [uid,iid,'cnt']
# train2 = train2.merge(tmp, on = [uid,iid], how='left')
# train2 = train2.sort_values([time_col, 'cnt'],ascending=False)
# train2.index = range(len(train2))
# positive_items_per_user2 = train2.groupby([uid])[iid].apply(list)


# tmp = train3.groupby([uid,iid])[time_col].agg('count').reset_index()
# tmp.columns = [uid,iid,'cnt']
# train3 = train3.merge(tmp, on = [uid,iid], how='left')
# train3 = train3.sort_values([time_col, 'cnt'],ascending=False)
# train3.index = range(len(train3))
# positive_items_per_user3 = train3.groupby([uid])[iid].apply(list)


# tmp = train4.groupby([uid,iid])[time_col].agg('count').reset_index()
# tmp.columns = [uid,iid,'cnt']
# train4 = train4.merge(tmp, on = [uid,iid], how='left')
# train4 = train4.sort_values([time_col, 'cnt'],ascending=False)
# train4.index = range(len(train4))
# positive_items_per_user4 = train4.groupby([uid])[iid].apply(list)
# sub = pd.read_csv('../../data/sample_submission.csv')
# result = []

# userindexes = {svd.users[i]:i for i in range(len(svd.users))}
# for user in tqdm(sub[uid].unique()):
#     user_output = []
#     if user in positive_items_per_user1.keys():
#         most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
#         user_index = userindexes[user]
#         new_order = {}
#         for k in list(most_common_items_of_user.keys())[:20]:
#             try:
#                 itemindex = svd.items.index(k)
#                 pred_value = np.dot(svd.userfeatures[user_index], svd.itemfeatures[itemindex].T) + svd.item_bias[0, itemindex]
#             except:
#                 pred_value = most_common_items_of_user[k]
#             new_order[k] = pred_value
#         user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:20]
        
#     elif user in positive_items_per_user2.keys():
#         most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
#         user_index = userindexes[user]
#         new_order = {}
#         for k in list(most_common_items_of_user.keys())[:20]:
#             try:
#                 itemindex = svd.items.index(k)
#                 pred_value = np.dot(svd.userfeatures[user_index], svd.itemfeatures[itemindex].T) + svd.item_bias[0, itemindex]
#             except:
#                 pred_value = most_common_items_of_user[k]
#             new_order[k] = pred_value
#         user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:20]
        
#     elif user in positive_items_per_user3.keys():
#         most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
#         user_index = userindexes[user]
#         new_order = {}
#         for k in list(most_common_items_of_user.keys())[:20]:
#             try:
#                 itemindex = svd.items.index(k)
#                 pred_value = np.dot(svd.userfeatures[user_index], svd.itemfeatures[itemindex].T) + svd.item_bias[0, itemindex]
#             except:
#                 pred_value = most_common_items_of_user[k]
#             new_order[k] = pred_value
#         user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:20]
        
#     elif user in positive_items_per_user4.keys():
#         most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
#         user_index = userindexes[user]
#         new_order = {}
#         for k in list(most_common_items_of_user.keys())[:20]:
#             try:
#                 itemindex = svd.items.index(k)
#                 pred_value = np.dot(svd.userfeatures[user_index], svd.itemfeatures[itemindex].T) + svd.item_bias[0, itemindex]
#             except:
#                 pred_value = most_common_items_of_user[k]
#             new_order[k] = pred_value
#         user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:20]
    
#     if user in user_group_dict:
#         item_his = user_group_dict[user][::-1]
#         for item in item_his:
#             if item in pred_next and pred_next[item] not in user_output:
#                 user_output += [pred_next[item]]
#     if len(user_output) > 20:
#         user_output = user_output[:20]
        
#     if len(user_output) < 20:
#         user_output += list(popular_items[:20 - len(user_output)])
    
#     assert(len(user_output) == 20) 
#     user_output = ' '.join(user_output)
#     result.append([user, user_output])

# result = pd.DataFrame(result)
# result.columns = [uid, 'prediction']
# result.to_csv("./candidates/candidates_recall_20.csv", index=False)

import os
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

# for col in users.columns:
#     if col in categories:
#         users[col] = pd.Series(pd.Categorical(users[col], dtype=categories[col]))

# for col in items.columns:
#     if col in categories:
#         items[col] = pd.Series(pd.Categorical(items[col], dtype=categories[col]))

items_features = pd.read_csv('./data/item_features_new.csv')
items_features['article_id'] = items_features['article_id'].astype('str')
items = items.merge(items_features, on='article_id')

for key in scalers:
    if key in items.columns:
        items[key] = (items[key] - scalers[key][0]) / scalers[key][1]
    if key in users.columns:
        users[key] = (users[key] - scalers[key][0]) / scalers[key][1]

encoder = OrdinalEncoder().fit(users['customer_id'].to_numpy().reshape(-1, 1))
users['customer_id'] = encoder.transform(users['customer_id'].to_numpy().reshape(-1, 1)).flatten()

items = items.drop(columns=['prod_name', 'detail_desc'])

prediction_date = pd.Series(['2020-09-22'])
prediction_date = pd.to_datetime(prediction_date, format=DATETIME_FORMAT)
prediction_date = prediction_date.view('int64') // 10 ** 9
prediction_date = prediction_date.values[0]
prediction_date = (prediction_date - scalers['timestamp'][0]) / scalers['timestamp'][1]
users['timestamp'] = prediction_date

candidate_files = os.listdir('./candidates')
c = pd.DataFrame()
for file in candidate_files:
    if file != 'candidates_recall.csv':
        candidates = pd.read_csv(f'./candidates/{file}')
        print(f'loaded {file}...')
        c = c.append(candidates)

overall_unique_candidates = set()
c = c.groupby('customer_id')['prediction'].agg(lambda x: ' '.join(list(x))).reset_index()
c_dict = c.to_dict()
c_dict_by_users = {}
id, candidates = tuple(c_dict.keys())
for i in tqdm(range(len(c_dict[id].keys()))):
    c_dict_by_users[c_dict[id][i]] = set(map(lambda x: int(x), c_dict[candidates][i].replace(',', '').replace('[', '').replace(']', '').split(' ')))
    overall_unique_candidates |= c_dict_by_users[c_dict[id][i]]
overall_unique_candidates = list(map(str, overall_unique_candidates))
# print()
# for id in ids:
#     print(id, len(c_dict_by_users[id]), c_dict_by_users[id], end='\n\n')

BATCH_SIZE = 200
items = items[items['article_id'].isin(overall_unique_candidates)].reset_index().drop(columns=['index'])
print(f'{items.shape[0]} unique items to consider in total...')

del categories['sales_channel_id']

write_interval, j, counter = 1000, 0, 0
predictions = pd.DataFrame()
with tqdm(total=users.shape[0]) as progress:
    for i in range(0, users.shape[0], BATCH_SIZE):
        user_batch = users.loc[i:i+BATCH_SIZE-1].reset_index().drop(columns=['index'])
        batch_data = pd.DataFrame()
        uids = user_batch['customer_id'].to_list()
        uids_inv = encoder.inverse_transform([[uid] for uid in uids]).flatten()
        sample_iids = [list(map(str, c_dict_by_users[uid])) for uid in uids_inv]
        batch_unique_candidates = set()
        for iid in sample_iids:
            batch_unique_candidates |= set(iid)
        unique_batch_items = items[items['article_id'].isin(batch_unique_candidates)].reset_index().drop(columns=['index'])
        for uid, iids in zip(uids, sample_iids):
            item_sample = unique_batch_items[unique_batch_items['article_id'].isin(iids)].reset_index().drop(columns=['index'])
            user_sample = user_batch[user_batch['customer_id'] == uid].reset_index().drop(columns=['index'])
            num_items = item_sample.shape[0]
            data_dict = item_sample.to_dict(orient='list')
            for col in user_sample.columns:
                data_dict[col] = user_sample[col].to_list() * num_items
            batch_data = batch_data.append(pd.DataFrame(data_dict))
        batch_data = batch_data.reset_index().drop(columns=['index'])
        batch_data['article_id'] = batch_data['article_id'].astype('int64')
        for col in batch_data.columns:
            if col in categories:
                batch_data[col] = pd.Series(pd.Categorical(batch_data[col], dtype=categories[col]))
        # if i == 0:
        #     print(batch_data.head())
        #     batch_data.info()
        #     break
        batch_pred = booster.predict(batch_data)
        predictions = predictions.append(pd.DataFrame({
            'customer_id': batch_data['customer_id'].to_numpy(),
            'article_id': batch_data['article_id'].to_numpy(),
            'rank': batch_pred
        }))
        progress.update(user_batch.shape[0])
        if j < write_interval:
            j += 1
        else:
            predictions.reset_index().drop(columns=['index']).to_csv(f'./submissions/master_04_21_22/master_{counter}_c25rcpop.csv', index=False)
            predictions = pd.DataFrame()
            j = 0
            tqdm.write(f'saved predictions to master_{counter}_c25rcpop.csv...')
            counter += 1

if 0 < j <= write_interval:
    predictions.reset_index().drop(columns=['index']).to_csv(f'./submissions/master_04_21_22/master_{counter}_c25rcpop.csv', index=False)
    print(f'saved predictions to master_{counter}_c25rcpop.csv...')

# predictions.reset_index().drop(columns=['index']).to_csv('./submissions/master_04_21_22/candidate_25_recall_popularity.csv')


