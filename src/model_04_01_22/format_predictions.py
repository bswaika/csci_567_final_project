import pandas as pd

print('Loading...')

predictions = pd.read_csv('../../dist/model_04_01_22/pre_submit.csv')
predictions = predictions[['customer_id', 'article_id', 'rank']]

popular = pd.read_csv('../../dist/model_04_01_22/popular_items.csv')
popular = popular.loc[:1, ['article_id', 'total']]

print('Loaded...')

predictions = predictions.groupby('customer_id')[['article_id']].agg(lambda x: list(x)).reset_index()
predictions['article_id'] = predictions['article_id'].apply(lambda x: ' '.join(map(str, x + popular['article_id'].to_list())))

predictions = predictions.rename(columns={'article_id': 'prediction'})

print('Writing...')

predictions.to_csv('../../dist/model_04_01_22/submission.csv', index=False)

print('Done...')