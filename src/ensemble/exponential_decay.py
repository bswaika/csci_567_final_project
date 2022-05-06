import numpy as np
import pandas as pd
import os
import glob
#import reco
from tqdm import tqdm
import datetime
import gc
import random

data = pd.read_csv("../../data/transactions_train.csv", dtype={'article_id':str})

print("All Transactions Date Range: {} to {}".format(data['t_dat'].min(), data['t_dat'].max()))

data["t_dat"] = pd.to_datetime(data["t_dat"])
train1 = data.loc[(data["t_dat"] >= datetime.datetime(2020,9,8)) & (data['t_dat'] < datetime.datetime(2020,9,16))]
train2 = data.loc[(data["t_dat"] >= datetime.datetime(2020,9,1)) & (data['t_dat'] < datetime.datetime(2020,9,8))]
train3 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,23)) & (data['t_dat'] < datetime.datetime(2020,9,1))]
train4 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,15)) & (data['t_dat'] < datetime.datetime(2020,8,23))]
train5 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,7)) & (data['t_dat'] < datetime.datetime(2020,8,15))]

val = data.loc[data["t_dat"] >= datetime.datetime(2020,9,16)]

articles_df = pd.read_csv('../../data/articles.csv') #, dtype={'article_id': str})

def get_alternate_most_popular(df_data, factor, return_orig=False):
    
    next_best_match = []
    
    df = df_data.copy()
    df['article_count'] = df.groupby('article_id')['customer_id'].transform('count')
    df['article_min_price'] = df.groupby('article_id')['price'].transform('min')
    count_df = df[['article_id', 'article_count', 'article_min_price']].drop_duplicates().reset_index(drop=True)
    
    del df
    
    for article in tqdm(count_df.article_id.tolist()):
        prodname = articles_df[articles_df.article_id==int(article)]['prod_name'].iloc[0]
        other_article_list = articles_df[articles_df.prod_name==prodname]['article_id'].tolist()
        other_article_list.remove(int(article))
        k = len(other_article_list)
        if k==1:
            next_best_match.append(other_article_list[0])
        if k>1:
            if len(count_df[np.in1d(count_df['article_id'], other_article_list)])!=0:
                next_best_match.append(count_df[np.in1d(count_df['article_id'], other_article_list)].sort_values('article_count', ascending=False)['article_id'].iloc[0])
            else:
                next_best_match.append(np.nan)
        if k==0:
            next_best_match.append(np.nan)

    count_df['next_best_article'] = next_best_match
    count_df['next_best_article'] = count_df['next_best_article'].fillna(0).astype(int)
    count_df['next_best_article'] = np.where(count_df['next_best_article']==0, count_df['article_id'], str(0)+count_df['next_best_article'].astype(str))

    right_df = count_df[['next_best_article']].copy().rename(columns={'next_best_article':'article_id'})

    next_best_count = []
    next_best_price = []
    for article in tqdm(right_df['article_id']):
        if len(count_df[count_df.article_id==article]['article_count'])>0:
            next_best_count.append(count_df[count_df.article_id==article]['article_count'].iloc[0])
            next_best_price.append(count_df[count_df.article_id==article]['article_min_price'].iloc[0])
        else:
            next_best_count.append(0)
            next_best_price.append(0)

    count_df['count_next_best'] = next_best_count
    count_df['next_best_min_price'] = next_best_price
        
    more_popular_alternatives = count_df[(count_df.article_min_price >= count_df.next_best_min_price) & 
                                         (count_df.count_next_best > factor *count_df.article_count)].copy().reset_index(drop=True)
    more_popular_alt_list = more_popular_alternatives.article_id.unique().tolist()
    
    if return_orig:
        return more_popular_alt_list, more_popular_alternatives, count_df
    else:
        return more_popular_alt_list, more_popular_alternatives

alt_list_1v, alt_df_1v = get_alternate_most_popular(train2, 2, return_orig=False)
alt_list_2v, alt_df_2v = get_alternate_most_popular(train3, 2, return_orig=False)
alt_list_3v, alt_df_3v = get_alternate_most_popular(train4, 2, return_orig=False)
alt_list_4v, alt_df_4v = get_alternate_most_popular(train5, 2, return_orig=False)

positive_items_per_user1 = train1.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user2 = train2.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user3 = train3.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user4 = train4.groupby(['customer_id'])['article_id'].apply(list)

train = pd.concat([train1, train2], axis=0)
train['pop_factor'] = train['t_dat'].apply(lambda x: 1/(datetime.datetime(2020,9,16) - x).days)
popular_items_group = train.groupby(['article_id'])['pop_factor'].sum()

_, popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])

def apk(actual, predicted, k=12):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=12):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

positive_items_val = val.groupby(['customer_id'])['article_id'].apply(list)

val_users = positive_items_val.keys()
val_items = []

for i,user in tqdm(enumerate(val_users)):
    val_items.append(positive_items_val[user])
    
print("Total users in validation:", len(val_users))

from collections import Counter
outputs = []
cnt = 0

popular_items = list(popular_items)

for user in tqdm(val_users):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_1v:
                al.append(alt_df_1v[alt_df_1v.article_id==l[j]]['next_best_article'].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_2v:
                al.append(alt_df_2v[alt_df_2v.article_id==l[j]]['next_best_article'].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_3v:
                al.append(alt_df_3v[alt_df_3v.article_id==l[j]]['next_best_article'].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_4v:
                al.append(alt_df_4v[alt_df_4v.article_id==l[j]]['next_best_article'].iloc[0])
        l = l + al
        user_output += l[:12]
    
    user_output += list(popular_items[:12 - len(user_output)])    
    outputs.append(user_output)
    
print("mAP Score on Validation set:", mapk(val_items, outputs))

train1 = data.loc[(data["t_dat"] >= datetime.datetime(2020,9,16)) & (data['t_dat'] < datetime.datetime(2020,9,23))]
train2 = data.loc[(data["t_dat"] >= datetime.datetime(2020,9,8)) & (data['t_dat'] < datetime.datetime(2020,9,16))]
train3 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,31)) & (data['t_dat'] < datetime.datetime(2020,9,8))]
train4 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,23)) & (data['t_dat'] < datetime.datetime(2020,8,31))]
train5 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,15)) & (data['t_dat'] < datetime.datetime(2020,8,23))]

alt_list_1, alt_df_1 = get_alternate_most_popular(train2, 2, return_orig=False)
alt_list_2, alt_df_2 = alt_list_1v, alt_df_1v
alt_list_3, alt_df_3 = alt_list_2v, alt_df_2v
alt_list_4, alt_df_4 = alt_list_3v, alt_df_3v

positive_items_per_user1 = train1.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user2 = train2.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user3 = train3.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user4 = train4.groupby(['customer_id'])['article_id'].apply(list)

train = pd.concat([train1, train2], axis=0)
train['pop_factor'] = train['t_dat'].apply(lambda x: 1/(datetime.datetime(2020,9,23) - x).days)
popular_items_group = train.groupby(['article_id'])['pop_factor'].sum()

_, popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])

user_group = pd.concat([train1, train2, train3, train4], axis=0).groupby(['customer_id'])['article_id'].apply(list)
submission = pd.read_csv("../../data/sample_submission.csv")

from collections import Counter
outputs = []
cnt = 0

for user in tqdm(submission['customer_id']):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_1:
                al.append(alt_df_1[alt_df_1.article_id==l[j]]['next_best_article'].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_2:
                al.append(alt_df_2[alt_df_2.article_id==l[j]]['next_best_article'].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_3:
                al.append(alt_df_3[alt_df_3.article_id==l[j]]['next_best_article'].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_4:
                al.append(alt_df_4[alt_df_4.article_id==l[j]]['next_best_article'].iloc[0])
        l = l + al        
        user_output += l[:12]
    
    user_output += list(popular_items[:12 - len(user_output)])
    outputs.append(user_output)
    
str_outputs = []
for output in outputs:
    str_outputs.append(" ".join([str(x) for x in output]))

submission['prediction'] = str_outputs
submission.to_csv("./submissions/exponential_decay.csv", index=False)