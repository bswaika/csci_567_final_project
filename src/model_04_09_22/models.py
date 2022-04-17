'''
    Module
    ------
    Currently a script to implement models. Will change later
    into a module to contain only model classes
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import uuid, os

from constants import DATA_FILES, DATA_SOURCE_DIR, MODEL_DIR, NUM_ITEMS, MODEL_TRACKER
import transactions

FEATURES_DIR = f'{DATA_SOURCE_DIR}/features'
LABELS_DIR = f'{DATA_SOURCE_DIR}/labels'

# FILE_ID = '7cb8b45d-d4b9-44d8-ba51-554ab92a0222'
FILE_ID = 'ff01743d-cb89-4ffe-8e88-0895218dea7f'

features = pd.read_csv(f'{FEATURES_DIR}/{FILE_ID}.csv')
labels = pd.read_csv(f'{LABELS_DIR}/{FILE_ID}.csv')

features = transactions.set_date_column_type(features)

# Helper Functions -- Move this later to a better module
convert_str_to_int_list = lambda df: df.apply(lambda x: list(map(int, x.replace('[', '').replace(']', '').split(', '))))
convert_df_to_dict = lambda df: {col: df[col].to_numpy() for col in df.columns}

def get_trainanble_data(df: pd.DataFrame):
    df['t_dat'] = df['t_dat'].astype(np.int64) // 10 ** 9
    df['id'] = df['id'].apply(lambda x: x[:16])
    max_len = df.groupby('id')['customer_id'].count().reset_index()['customer_id'].max()
    df = df.groupby('id').agg(lambda x: list(x)).reset_index().drop(columns='id')
    df = convert_df_to_dict(df)
    data = {k: np.array(list(map(lambda x: np.pad(np.array(x), (max_len-len(x), 0)), df[k].tolist()))) for k in df}
    print(f'SEQ: {max_len}')
    return data

def get_encoded_labels(labels: pd.Series):
    labels = convert_str_to_int_list(labels)
    items = pd.read_csv(DATA_FILES['items'])
    id_lookup = tf.keras.layers.IntegerLookup(max_tokens=NUM_ITEMS+1, output_mode='count', pad_to_max_tokens=True, mask_token=0)
    id_lookup.adapt(items['article_id'].to_numpy())
    max_len = labels.apply(lambda x: len(x)).max()
    data = np.array(list(map(lambda x: np.pad(np.array(x), (max_len-len(x), 0)), labels)))
    data = id_lookup(data)
    sums = tf.reduce_sum(data, axis=1, keepdims=True)
    data = data / sums
    print(f'LAB: {max_len}')
    return data, id_lookup.get_vocabulary()

def save_model(id, loss):
    with open(MODEL_TRACKER, 'a') as outfile:
        outfile.write(f'{id},{loss}\n')

# Preprocessing
# users = pd.read_csv(DATA_FILES['users'])
# items = pd.read_csv(DATA_FILES['items'])

# print(users.shape, items.shape)

# df_dict = convert_df_to_dict(users.loc[:20, :])
# for key in df_dict:
#     print(key, ':', df_dict[key])


class UserEmbedding(tf.keras.Model):
    def __init__(self, oov, embed_dim, **kwargs):
        super().__init__(**kwargs)
        users = pd.read_csv(DATA_FILES['users'])
        self._id_lookup = tf.keras.layers.StringLookup(num_oov_indices=oov, mask_token='0')
        self._id_lookup.adapt(users['customer_id'].to_numpy())

        self._age_normalizer = tf.keras.layers.Normalization(axis=None)
        self._age_normalizer.adapt(users['age'].to_numpy())

        self._membership_lookup = tf.keras.layers.StringLookup(mask_token='0')
        self._membership_lookup.adapt(users['club_member_status'].to_numpy())

        self._zip_lookup = tf.keras.layers.StringLookup(mask_token='0')
        self._zip_lookup.adapt(users['postal_code'].to_numpy())

        self._id = tf.keras.Sequential([
            tf.keras.Input(shape=(None,), dtype=tf.string),
            self._id_lookup,
            tf.keras.layers.Embedding(len(self._id_lookup.get_vocabulary()), embed_dim)
        ])

        self._membership = tf.keras.Sequential([
            tf.keras.Input(shape=(None,), dtype=tf.string),
            self._membership_lookup,
            tf.keras.layers.Embedding(len(self._membership_lookup.get_vocabulary()), embed_dim)
        ])

        self._zip = tf.keras.Sequential([
            tf.keras.Input(shape=(None,), dtype=tf.string),
            self._zip_lookup,
            tf.keras.layers.Embedding(len(self._zip_lookup.get_vocabulary()), embed_dim)
        ])        

        del users
    
    def call(self, inputs):
        return tf.concat([
            self._id(inputs['customer_id']),
            self._membership(inputs['club_member_status']),
            self._zip(inputs['postal_code']),
            tf.reshape(self._age_normalizer(inputs['age']), (*inputs['age'].shape, 1))
        ], axis=-1)

class ItemEmbedding(tf.keras.Model):
    def __init__(self, oov, embed_dim, **kwargs):
        super().__init__(**kwargs)
        items = pd.read_csv(DATA_FILES['items'])

        max_len_kw = max(map(lambda x: len(x.split()), items['keywords'].to_numpy()))
        max_len_pn = max(map(lambda x: len(x.split()), items['prod_name'].to_numpy()))

        self._id_lookup = tf.keras.layers.IntegerLookup(num_oov_indices=oov, mask_token=0)
        self._id_lookup.adapt(items['article_id'].to_numpy())

        self._keyword_vectorizer = tf.keras.layers.TextVectorization(standardize='lower_and_strip_punctuation', split='whitespace', output_sequence_length=max_len_kw)
        self._keyword_vectorizer.adapt(items['keywords'].to_numpy())

        self._name_vectorizer = tf.keras.layers.TextVectorization(standardize='lower_and_strip_punctuation', split='whitespace', output_sequence_length=max_len_pn)
        self._name_vectorizer.adapt(items['prod_name'].to_numpy())

        self._id = tf.keras.Sequential([
            tf.keras.Input(shape=(None,), dtype=tf.int64),
            self._id_lookup,
            tf.keras.layers.Embedding(len(self._id_lookup.get_vocabulary()), embed_dim)
        ])

        self._keyword = tf.keras.Sequential([
            tf.keras.Input(shape=(None, 1), dtype=tf.string),
            self._keyword_vectorizer,
            tf.keras.layers.Embedding(len(self._keyword_vectorizer.get_vocabulary()), embed_dim, mask_zero=True),
        ])

        self._prod_name = tf.keras.Sequential([
            tf.keras.Input(shape=(None, 1), dtype=tf.string),
            self._name_vectorizer,
            tf.keras.layers.Embedding(len(self._name_vectorizer.get_vocabulary()), embed_dim, mask_zero=True),
        ])

    def call(self, inputs):
        return tf.concat([
            self._id(inputs['article_id']),
            tf.reduce_mean(self._keyword(inputs['keywords']), axis=-2),
            tf.reduce_mean(self._prod_name(inputs['prod_name']), axis=-2)
        ], axis=-1)

class TimeSeriesModel(tf.keras.Model):
    def __init__(self, timestamps, **kwargs):
        super().__init__(**kwargs)
        self.user = UserEmbedding(10, 8)
        self.item = ItemEmbedding(10, 8)
        self.time = tf.keras.layers.Normalization(axis=None)
        self.time.adapt((timestamps.astype(np.int64) // 10 ** 9).to_numpy())
        self.lstm = tf.keras.layers.LSTM(1024)
        self.dense = tf.keras.layers.Dense(501, activation='softmax')
    
    def call(self, inputs):
        user = self.user(inputs)
        item = self.item(inputs)
        time = tf.reshape(self.time(inputs['t_dat']), (*inputs['t_dat'].shape, 1))
        features = tf.concat([user, item, time], axis=-1)
        outputs = self.lstm(features)
        outputs = self.dense(outputs)
        return outputs


# Test on a subset
# test_features = features.loc[:100, :]
# test_features['t_dat'] = test_features['t_dat'].astype(np.int64) // 10 ** 9
# test_features['id'] = test_features['id'].apply(lambda x: x[:16])
# max_len = test_features.groupby('id')['customer_id'].count().reset_index()['customer_id'].max()
# test_features = test_features.groupby('id').agg(lambda x: list(x)).reset_index().drop(columns='id')

# # print(ages)
# # print(max_len)
# # print(test_features)

# test_features = convert_df_to_dict(test_features)

# trial = {k: np.array(list(map(lambda x: np.pad(np.array(x), (max_len-len(x), 0)), test_features[k].tolist()))) for k in test_features}

# test_features = get_trainanble_data(test_features)

# print(trial['age'])
# print(trial['customer_id'])
# print(trial['t_dat'])
# print(trial['article_id'])

# user_model = UserEmbedding(10, 8)
# item_model = ItemEmbedding(10, 8)
# user_embeddings = user_model(test_features)
# item_embeddings = item_model(test_features)
# print(user_embeddings[0])
# print(item_embeddings[0])

# print(encode_labels(labels.loc[:5, 'labels']))

X = get_trainanble_data(features)
Y, Y_vocab = get_encoded_labels(labels['labels'])

train_size = int(X['t_dat'].shape[0] * 0.8)

print(X['t_dat'].shape, Y.shape)

X_train, X_test = {k : X[k][:train_size] for k in X}, {k : X[k][train_size:] for k in X}
Y_train, Y_test = Y[:train_size], Y[train_size:]

EPOCHS = 300
BATCH_SIZE = 50

# 2
model = TimeSeriesModel(features['t_dat'])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adadelta(), metrics=[tf.keras.metrics.MeanSquaredError()])
model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
loss, mse = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
MODEL_ID = f'model_{uuid.uuid4()}_{FILE_ID}'
os.mkdir(f'{MODEL_DIR}/{MODEL_ID}')
model.save_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
save_model(MODEL_ID, loss)
predictions = model(X_test)
print(loss, mse)

with open(f'{MODEL_DIR}/{MODEL_ID}/predictions.csv', 'w') as outfile:
    for prediction in predictions:
        sorted_prediction = sorted(range(len(Y_vocab)), key=lambda x: prediction[x], reverse=True)
        outfile.write(','.join(map(lambda x: str(Y_vocab[x]), sorted_prediction[:10])) + '\n')

# 3
model = TimeSeriesModel(features['t_dat'])
model.compile(loss=tf.keras.losses.KLDivergence(), optimizer=tf.keras.optimizers.Adadelta(), metrics=[tf.keras.metrics.MeanSquaredError()])
model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
loss, mse = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
MODEL_ID = f'model_{uuid.uuid4()}_{FILE_ID}'
os.mkdir(f'{MODEL_DIR}/{MODEL_ID}')
model.save_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
save_model(MODEL_ID, loss)
predictions = model(X_test)
print(loss, mse)

with open(f'{MODEL_DIR}/{MODEL_ID}/predictions.csv', 'w') as outfile:
    for prediction in predictions:
        sorted_prediction = sorted(range(len(Y_vocab)), key=lambda x: prediction[x], reverse=True)
        outfile.write(','.join(map(lambda x: str(Y_vocab[x]), sorted_prediction[:10])) + '\n')