import pandas as pd
from load import load_transactions

iid = 'article_id'
uid = 'customer_id'
t = 'timestamp'

transactions = load_transactions()
current_date = transactions[t].max()
item_last_sold = transactions.groupby(iid)[t].max().reset_index()
recent_items = item_last_sold[item_last_sold[t] >= current_date - 60*60*24*7*4*16]
print(recent_items.shape)