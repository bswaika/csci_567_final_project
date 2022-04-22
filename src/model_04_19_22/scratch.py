from load import load_transactions, load_items, load_users
import pandas as pd

from tqdm import tqdm

items = load_items()
transactions = load_transactions(False)
users = load_users()

# transactions = transactions.loc[:200000]
# users = users[users['customer_id'].isin(transactions['customer_id'].unique())]

# Extract average price of an item in the sample
# item_price = transactions.groupby('article_id')['price'].mean().reset_index().rename(columns={'price': 'avg_price'})
# items = items.merge(item_price, on='article_id')
# transactions = transactions.drop(columns='price')

# # Extract popularity of an item in the sample
# item_popularity = transactions.groupby('article_id')['timestamp'].count().reset_index().rename(columns={'timestamp': 'popularity'})
# total_purchases = item_popularity['popularity'].sum()
# item_popularity['popularity'] = item_popularity['popularity'] / total_purchases
# items = items.merge(item_popularity, on='article_id')

# items = items[['article_id','avg_price','popularity']]
# print(items.head())
# items.to_csv('./data/item_features_new.csv', index=False)

# top_7k_items = transactions.groupby('article_id')['timestamp'].count().reset_index().rename(columns={'timestamp': 'count'}).sort_values(by='count', ascending=False).reset_index().loc[:6999, :].drop(columns=['index'])
# top_7k_items.to_csv('./data/top_7000_articles.csv', index=False)

users_unique_items = transactions.groupby('customer_id')['article_id'].agg(lambda x: list(set(x))).reset_index().rename(columns={'article_id':'unique_articles'})

with tqdm(total=users_unique_items.shape[0]) as pbar:
    def extract_categories(i):
        item_features = items[items['article_id'].isin(i)][['index_name', 'section_name', 'product_type_name', 'perceived_colour_value_name']]
        query = '|'.join([f'{col}:{",".join(item_features[col].unique().tolist())}' for col in item_features.columns])
        pbar.update(1)
        return query
    users_unique_items['query'] = users_unique_items['unique_articles'].apply(extract_categories)

users_unique_items.to_csv('./data/user_queries.csv', index=False)


# if __name__ == 'main':
#     mp.set_start_method('spawn', force=True)
#     manager = Manager()
#     threads, result = [], manager.list()
#     num_threads = 8
#     factor = users_unique_items.shape[0] // num_threads
#     for index in range(num_threads):
#         start, end = factor * index, factor * (index + 1)
#         thread = Process(target=generate_query, args=[users_unique_items.loc[start:end-1], items, result])
#         threads.append(thread)
#         thread.start()

#     generate_query(users_unique_items.loc[end:], result)

#     for thread in threads:
#         thread.join()

#     user_query = pd.DataFrame
#     for df in result:
#         user_query = user_query.append(df)

#     print(user_query.head())
#     user_query.to_csv('./data/user_queries.csv', index=False)

