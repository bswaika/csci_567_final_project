import os
import gc
import tensorflow as tf
from threading import Thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from articles import load_articles, make_article_preprocessing_model, query_articles
from customers import load_customers, make_customer_preprocessing_model, query_customers
from transactions import load_transactions, train_test_split, transform_transactions
from model import RecSysModel

print('started...')
print('loading articles...')
articles, ia, k, m = load_articles('./data/articles_keywords.csv')
print(f'{articles[0].shape[0]} articles loaded...')
print('loading customers...')
customers, ic = load_customers('./data/customers.csv')
print(f'{customers[0].shape[0]} customers loaded...')
print('loading transactions...')
transactions = load_transactions('./data/transactions_popular.csv')
print(f'{transactions.shape[0]} transactions loaded...')

gc.collect()

print('processing transactions...')
transactions, validate = train_test_split(transactions, 7)
transactions = transform_transactions(transactions)
validate = transform_transactions(validate)
print(f'{transactions[0].shape[0]} transactions processed...')

gc.collect()

print('processing queries...')
customer_data, article_data = [], []
customer_thread = Thread(target=query_customers, args=[transactions[0][:, 0], customers, customer_data])
article_thread = Thread(target=query_articles, args=[transactions[1][:, 0], articles, article_data])
customer_thread.start()
article_thread.start()
article_thread.join()
customer_thread.join()

customer_val_data, article_val_data = [], []
customer_thread = Thread(target=query_customers, args=[validate[0][:, 0], customers, customer_val_data])
article_thread = Thread(target=query_articles, args=[validate[1][:, 0], articles, article_val_data])
customer_thread.start()
article_thread.start()
article_thread.join()
customer_thread.join()
print(f'{transactions[0].shape[0]} queries processed...')

print('starting tensorflow training...')
article_preproc_model = make_article_preprocessing_model(ia, k, m)
article_preproc_model.compile()
article_data = article_preproc_model.predict(article_data)
article_val_data = article_preproc_model.predict(article_val_data)
customer_preproc_model = make_customer_preprocessing_model(ic)
customer_preproc_model.compile()
customer_data = customer_preproc_model.predict(customer_data)
customer_val_data = customer_preproc_model.predict(customer_val_data)
print('generated lookups...')

gc.collect()

dict_sizes = {
    'customers': len(customer_preproc_model.customer_dict),
    'articles_ids': len(article_preproc_model.article_dict),
    'articles_kws': len(article_preproc_model.keyword_dict)
}

embed_dims = {
    'customers': 4,
    'articles': 4
}

dense_params = {
    'customers': [3, 100, 0.5, 4],
    'articles': [3, 100, 0.5, 4],
    'similarity': [4, 64, 0.5, 2]
}

EPOCHS, BATCH_SIZE = 50, 10_000

recommendation_model = RecSysModel(dict_sizes, embed_dims, dense_params, name='recommendation_model')
recommendation_model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss=tf.keras.losses.MeanSquaredError())
history = recommendation_model.fit([customer_data, article_data], [transactions[2], transactions[3]], epochs=EPOCHS, batch_size=BATCH_SIZE)
print('training complete...')
val_loss = recommendation_model.evaluate([customer_val_data, article_val_data], [validate[2], validate[3]], batch_size=BATCH_SIZE)
print('evaluation complete...')
preds = recommendation_model.predict([customer_val_data, article_val_data])
print('prediction complete...')

# SAVE LOGIC
print('saving model...')
recommendation_model.save('./models/model-r2')
print('writing losses...')
with open('./models/losses-r2.txt', 'w') as outfile:
    for loss in history.history['loss']:
        outfile.write(str(loss) + '\n')
    outfile.write('\n\n' + str(val_loss))
print('writing predictions...')
with open('./models/preds-r2.csv', 'w') as outfile:
    outfile.write('freq,spend_freq\n')
    for pred in preds:
        outfile.write(','.join(map(str, pred)) + '\n')

gc.collect()

print('done...')