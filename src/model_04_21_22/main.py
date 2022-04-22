import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from reco.recommender import FunkSVD
from reco.metrics import rmse
import datetime
from collections import Counter
from datetime import timedelta

train = pd.read_csv('../../data/transactions_train.csv', dtype={'article_id':str})

uid = 'customer_id'
iid = 'article_id'
time_col = 't_dat'

train[time_col] = pd.to_datetime(train[time_col])

one_day = timedelta(days=1)
year_month_day = str(train[time_col].max() - one_day*7*8)[:10].split('-')

year_ = int(year_month_day[0])
month_ = int(year_month_day[1])
day_ = int(year_month_day[2])

train_pop = train.loc[train[time_col] >= datetime.datetime(year_, month_, day_)]

train_pop['diff'] = (train_pop[time_col].max() - train_pop[time_col])
train_pop['pop_factor'] = 1 / (train_pop['diff'].dt.days + 1)

popular_items_group = train_pop.groupby([iid])['pop_factor'].sum()
_, popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])

def get_most_freq_next_item(user_group):
    next_items = {}
    for user in tqdm(user_group.keys()):
        items = user_group[user]
        for i,item in enumerate(items[:-1]):
            if item not in next_items:
                next_items[item] = []
#             if item != items[i+1]:
#                 next_items[item].append(items[i+1])
            next_items[item].append(items[i+1])
    
    pred_next = {}
    for item in next_items:
        if len(next_items[item]) >= 5:
            most_common = Counter(next_items[item]).most_common()
            ratio = most_common[0][1]/len(next_items[item])
            if ratio >= 0.1:
                pred_next[item] = most_common[0][0]
            
    return pred_next

one_day = timedelta(days=1)
year_month_day = str(train[time_col].max() - one_day*7*4)[:10].split('-')

year_ = int(year_month_day[0])
month_ = int(year_month_day[1])
day_ = int(year_month_day[2])

user_group = train.loc[train[time_col] >= datetime.datetime(year_, month_, day_)].groupby([uid])[iid].apply(list)
pred_next = get_most_freq_next_item(user_group)
user_group_dict = user_group.to_dict()

one_day = timedelta(days=1)
year_month_day = str(train[time_col].max() - one_day*7*16)[:10].split('-')

year_ = int(year_month_day[0])
month_ = int(year_month_day[1])
day_ = int(year_month_day[2])

df = train.loc[train[time_col] >= datetime.datetime(year_, month_, day_), [uid, iid, time_col]].copy()

df['diff'] = (df[time_col].max() - df[time_col])
df['pop_factor'] = 1 / (df['diff'].dt.days + 1)

popular_items_group = df.groupby([iid])['pop_factor'].sum()

# purchase count of each article
items_total_count = df.groupby([iid])[iid].count()
# purchase count of each user
users_total_count = df.groupby([uid])[uid].count()

df['feedback'] = 1
df = df.groupby([uid, iid]).sum().reset_index()
df['feedback'] = df.apply(lambda row: row['feedback']/popular_items_group[row[iid]], axis=1)

df['feedback'] = df['feedback'].apply(lambda x: 5.0 if x>5.0 else x)

df = df[[uid, iid, 'feedback']]

# shuffling
df = df.sample(frac=1).reset_index(drop=True)

svd = FunkSVD(k=8, learning_rate=0.01, regularizer=0.05, iterations=100, method='stochastic', bias=True)
svd.fit(X=df, formatizer={'user':uid, 'item':iid, 'value':'feedback'},verbose=True)

one_day = timedelta(days=1)

year_month_day_w1 = str(train[time_col].max() - one_day*7)[:10].split('-')
year_w1 = int(year_month_day_w1[0])
month_w1 = int(year_month_day_w1[1])
day_w1 = int(year_month_day_w1[2])
print(year_w1, month_w1, day_w1)


year_month_day_w2 = str(train[time_col].max() - one_day*7*2)[:10].split('-')
year_w2 = int(year_month_day_w2[0])
month_w2 = int(year_month_day_w2[1])
day_w2 = int(year_month_day_w2[2])
print(year_w2, month_w2, day_w2)


year_month_day_w3 = str(train[time_col].max() - one_day*7*3)[:10].split('-')
year_w3 = int(year_month_day_w3[0])
month_w3 = int(year_month_day_w3[1])
day_w3 = int(year_month_day_w3[2])
print(year_w3, month_w3, day_w3)


year_month_day_w4 = str(train[time_col].max() - one_day*7*4)[:10].split('-')
year_w4 = int(year_month_day_w4[0])
month_w4 = int(year_month_day_w4[1])
day_w4 = int(year_month_day_w4[2])
print(year_w4, month_w4, day_w4)

train1 = train.loc[(train[time_col] >= datetime.datetime(year_w1, month_w1, day_w1))]
train2 = train.loc[(train[time_col] >= datetime.datetime(year_w2, month_w2, day_w2)) & (train[time_col] < datetime.datetime(year_w1, month_w1, day_w1))]
train3 = train.loc[(train[time_col] >= datetime.datetime(year_w3, month_w3, day_w3)) & (train[time_col] < datetime.datetime(year_w2, month_w2, day_w2))]
train4 = train.loc[(train[time_col] >= datetime.datetime(year_w4, month_w4, day_w4)) & (train[time_col] < datetime.datetime(year_w3, month_w3, day_w3))]

tmp = train1.groupby([uid,iid])[time_col].agg('count').reset_index()
tmp.columns = [uid,iid,'cnt']
train1 = train1.merge(tmp, on = [uid,iid], how='left')
train1 = train1.sort_values([time_col, 'cnt'],ascending=False)
train1.index = range(len(train1))
positive_items_per_user1 = train1.groupby([uid])[iid].apply(list)


tmp = train2.groupby([uid,iid])[time_col].agg('count').reset_index()
tmp.columns = [uid,iid,'cnt']
train2 = train2.merge(tmp, on = [uid,iid], how='left')
train2 = train2.sort_values([time_col, 'cnt'],ascending=False)
train2.index = range(len(train2))
positive_items_per_user2 = train2.groupby([uid])[iid].apply(list)


tmp = train3.groupby([uid,iid])[time_col].agg('count').reset_index()
tmp.columns = [uid,iid,'cnt']
train3 = train3.merge(tmp, on = [uid,iid], how='left')
train3 = train3.sort_values([time_col, 'cnt'],ascending=False)
train3.index = range(len(train3))
positive_items_per_user3 = train3.groupby([uid])[iid].apply(list)


tmp = train4.groupby([uid,iid])[time_col].agg('count').reset_index()
tmp.columns = [uid,iid,'cnt']
train4 = train4.merge(tmp, on = [uid,iid], how='left')
train4 = train4.sort_values([time_col, 'cnt'],ascending=False)
train4.index = range(len(train4))
positive_items_per_user4 = train4.groupby([uid])[iid].apply(list)
sub = pd.read_csv('../../data/sample_submission.csv')
result = []

userindexes = {svd.users[i]:i for i in range(len(svd.users))}
for user in tqdm(sub[uid].unique()):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        user_index = userindexes[user]
        new_order = {}
        for k in list(most_common_items_of_user.keys())[:20]:
            try:
                itemindex = svd.items.index(k)
                pred_value = np.dot(svd.userfeatures[user_index], svd.itemfeatures[itemindex].T) + svd.item_bias[0, itemindex]
            except:
                pred_value = most_common_items_of_user[k]
            new_order[k] = pred_value
        user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:12]
        
    elif user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        user_index = userindexes[user]
        new_order = {}
        for k in list(most_common_items_of_user.keys())[:20]:
            try:
                itemindex = svd.items.index(k)
                pred_value = np.dot(svd.userfeatures[user_index], svd.itemfeatures[itemindex].T) + svd.item_bias[0, itemindex]
            except:
                pred_value = most_common_items_of_user[k]
            new_order[k] = pred_value
        user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:12]
        
    elif user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        user_index = userindexes[user]
        new_order = {}
        for k in list(most_common_items_of_user.keys())[:20]:
            try:
                itemindex = svd.items.index(k)
                pred_value = np.dot(svd.userfeatures[user_index], svd.itemfeatures[itemindex].T) + svd.item_bias[0, itemindex]
            except:
                pred_value = most_common_items_of_user[k]
            new_order[k] = pred_value
        user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:12]
        
    elif user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        user_index = userindexes[user]
        new_order = {}
        for k in list(most_common_items_of_user.keys())[:20]:
            try:
                itemindex = svd.items.index(k)
                pred_value = np.dot(svd.userfeatures[user_index], svd.itemfeatures[itemindex].T) + svd.item_bias[0, itemindex]
            except:
                pred_value = most_common_items_of_user[k]
            new_order[k] = pred_value
        user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:12]
    
    if user in user_group_dict:
        item_his = user_group_dict[user][::-1]
        for item in item_his:
            if item in pred_next and pred_next[item] not in user_output:
                user_output += [pred_next[item]]
    if len(user_output) > 12:
        user_output = user_output[:12]
        
    if len(user_output) < 12:
        user_output += list(popular_items[:12 - len(user_output)])
    
    assert(len(user_output) == 12) 
    user_output = ' '.join(user_output)
    result.append([user, user_output])

result = pd.DataFrame(result)
result.columns = [uid, 'prediction']
result.to_csv("./submissions.csv", index=False)
