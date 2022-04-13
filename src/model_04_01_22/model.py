####################################
###    Turicreate Recommender    ###
####################################

# https://medium.datadriveninvestor.com/how-to-build-a-recommendation-system-for-purchase-data-step-by-step-d6d7a78800b6

import data_parser.DataParser as dp
import turicreate as tc

# tc.config.set_num_gpus(1)

filepaths = {
    'articles': '../../data/articles.csv',
    'customers': '../../data/customers.csv',
    'transactions': '../../data/transactions_train.csv'
}

parser = dp.DataParser(filepaths)

articles = parser.get_data('articles')
customers = parser.get_data('customers')

# articles_similarity_model = tc.recommender.item_content_recommender.create(articles, item_id='article_id', max_item_neighborhood_size=32, similarity_metrics='cosine')
# articles_similarity_matrix = articles_similarity_model.get_similar_items(articles['article_id'], 12)

transactions = parser.get_data('transactions')
del parser
train = transactions.groupby(['customer_id', 'article_id'])[['t_dat']].count().reset_index().rename(columns={'t_dat': 'purchase_count'})
del transactions
train = train.merge(train.groupby(['customer_id'])['purchase_count'].sum().reset_index().rename(columns={'purchase_count': 'total'}), how='outer', on='customer_id')
train['weights'] = train['purchase_count'] / train['total']

print('Transforming to SFrame...')

train = tc.SFrame(data=train[['customer_id', 'article_id', 'weights']].to_dict(orient='list'))
articles = tc.SFrame(data=articles.to_dict(orient='list'))
customers = tc.SFrame(data=customers.to_dict(orient='list'))

print('Start Training...')

model = tc.recommender.item_similarity_recommender.create(train, user_id='customer_id', 
                                        item_id='article_id', target='weights', user_data=customers, 
                                        item_data=articles, only_top_k=12, similarity_type='cosine')
popular_items = transactions.groupby('article_id')['t_dat'].count().reset_index().rename(columns={'t_dat': 'purchase_count'}).sort_values('purchase_count', ascending=False).loc[:12, :]

predictions = model.recommend(customers['customer_id'], diversity=1, random_seed=9196)
