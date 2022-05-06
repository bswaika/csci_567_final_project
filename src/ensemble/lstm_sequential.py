import pandas as pd
df = pd.read_csv("../../data/articles.csv", dtype={'article_id': 'str'})

df = df.drop(columns = ['product_type_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name',
                        'perceived_colour_master_name', 'index_name', 'index_group_name', 'section_name', 
                        'garment_group_name', 'prod_name', 'department_name', 'detail_desc'])

temp = df.rename(
    columns={'article_id': 'item_id:token', 'product_code': 'product_code:token', 'product_type_no': 'product_type_no:float',
             'product_group_name': 'product_group_name:token_seq', 'graphical_appearance_no': 'graphical_appearance_no:token', 
             'colour_group_code': 'colour_group_code:token', 'perceived_colour_value_id': 'perceived_colour_value_id:token', 
             'perceived_colour_master_id': 'perceived_colour_master_id:token', 'department_no': 'department_no:token', 
             'index_code': 'index_code:token', 'index_group_no': 'index_group_no:token', 'section_no': 'section_no:token', 
             'garment_group_no': 'garment_group_no:token'})
temp.head()

temp.to_csv('./recbole/recbox_data.item', index=False, sep='\t')

df = pd.read_csv("../../data/transactions_train.csv", dtype={'article_id': 'str'})

df['t_dat'] = pd.to_datetime(df['t_dat'], format="%Y-%m-%d")

import numpy as np
df['timestamp'] = df.t_dat.values.astype(np.int64) // 10 ** 9

temp = df[df['timestamp'] > 1585620000][['customer_id', 'article_id', 'timestamp']].rename(
    columns={'customer_id': 'user_id:token', 'article_id': 'item_id:token', 'timestamp': 'timestamp:float'})

temp.to_csv('./recbole/recbox_data.inter', index=False, sep='\t')

import os
import numpy as np
import pandas as pd