'''
    Module
    ------
    Holds classes and helper functions to work with models and training
'''
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import DATA_FILES, MODEL_DIR, NUM_ITEMS, MODEL_TRACKER

# Helper Functions 
convert_str_to_int_list = lambda df: df.apply(lambda x: list(map(int, x.replace('[', '').replace(']', '').split(', '))))
convert_df_to_dict = lambda df: {col: df[col].to_numpy() for col in df.columns}
convert_time_to_int = lambda df: df.astype(np.int64) // 10 ** 9


def get_trainable_data(df: pd.DataFrame, max_len_threshold=45, mode='new'):
    if mode == 'old':
        df['id'] = df['id'].astype('str').apply(lambda x: x[:16])
    max_len = min(max_len_threshold, df.groupby('id')['customer_id'].count().reset_index()['customer_id'].max())
    print(f'SEQ: {max_len}')
    df = df.groupby('id').agg(lambda x: list(x)).reset_index().drop(columns='id')
    # sample_weights = df['customer_id'].apply(lambda x: ((max_len - len(x) + 1) if max_len <= max_len_threshold else 1) / (max_len + 1)).to_numpy()
    df = convert_df_to_dict(df)
    df = {k: np.array(list(map(lambda x: np.pad(np.array(x), (0, max_len-len(x))) if max_len - len(x) >= 0 else np.array(x[-max_len_threshold:]), df[k].tolist()))) for k in df.keys()}
    return df

def get_encoded_labels(labels: pd.Series):
    labels = convert_str_to_int_list(labels)
    items = pd.read_csv(DATA_FILES['items'])
    id_lookup = tf.keras.layers.IntegerLookup(max_tokens=NUM_ITEMS+1, output_mode='count', pad_to_max_tokens=True, mask_token=0)
    id_lookup.adapt(items['article_id'].to_numpy())
    max_len = labels.apply(lambda x: len(x)).max()
    labels = np.array(list(map(lambda x: np.pad(np.array(x), (max_len-len(x), 0)), labels)))
    labels = id_lookup(labels)
    sums = tf.reduce_sum(labels, axis=1, keepdims=True)
    labels = labels / sums
    print(f'LAB: {max_len}')
    return labels, id_lookup.get_vocabulary()

def save_model(id, loss):
    with open(MODEL_TRACKER, 'a') as outfile:
        outfile.write(f'{id},{loss}\n')

def train_test_split(X, Y, ratio):
    train_size = int(X['t_dat'].shape[0] * ratio)
    X_train, X_test = {k : X[k][:train_size] for k in X}, {k : X[k][train_size:] for k in X}
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    return (X_train, Y_train), (X_test, Y_test)

def generate_predictions(model, data, batch_size, Y_vocab):
    x = {k: data[k][:batch_size] for k in data}
    predictions = np.array(model(x))
    for i in tqdm(range(batch_size, data['t_dat'].shape[0], batch_size)):
        x = {k: data[k][i:i+batch_size] for k in data}
        predictions = np.append(predictions, model(x), axis=0)

    # predictions = np.array(predictions).reshape(data['t_dat'].shape[0], len(Y_vocab))

    return predictions

def save_predictions(predictions, MODEL_ID, Y_vocab):
    with open(f'{MODEL_DIR}/{MODEL_ID}/predictions.csv', 'w') as outfile:
        for prediction in tqdm(predictions):
            sorted_prediction = sorted(range(len(Y_vocab)), key=lambda x: prediction[x], reverse=True)
            outfile.write(','.join(map(lambda x: str(Y_vocab[x]), sorted_prediction[:10])) + '\n')


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
            tf.keras.layers.Embedding(len(self._id_lookup.get_vocabulary()), embed_dim, mask_zero=True)
        ])

        self._membership = tf.keras.Sequential([
            tf.keras.Input(shape=(None,), dtype=tf.string),
            self._membership_lookup,
            tf.keras.layers.Embedding(len(self._membership_lookup.get_vocabulary()), embed_dim, mask_zero=True)
        ])

        self._zip = tf.keras.Sequential([
            tf.keras.Input(shape=(None,), dtype=tf.string),
            self._zip_lookup,
            tf.keras.layers.Embedding(len(self._zip_lookup.get_vocabulary()), embed_dim, mask_zero=True)
        ])        

        del users
    
    def call(self, inputs):
        return tf.concat([
            self._id(inputs['customer_id']),
            self._membership(inputs['club_member_status']),
            self._zip(inputs['postal_code']),
            self._age_normalizer(inputs['age'])
            # tf.reshape(self._age_normalizer(inputs['age']), (*inputs['age'].shape, 1))
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
            tf.keras.layers.Embedding(len(self._id_lookup.get_vocabulary()), embed_dim, mask_zero=True)
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

        del items

    def call(self, inputs):
        return tf.concat([
            self._id(inputs['article_id']),
            tf.reduce_mean(self._keyword(inputs['keywords']), axis=-2),
            tf.reduce_mean(self._prod_name(inputs['prod_name']), axis=-2)
        ], axis=-1)

class UserModel(tf.keras.Model):
    def __init__(self, embed_dim, dense_config, **kwargs):
        '''
            Dense Config is an array of tuples containing (nodes, activation)
        '''
        super().__init__(**kwargs)
        self.embedding = UserEmbedding(10, embed_dim)
        self.dense_layers = [tf.keras.layers.Dense(nodes, activation=fn, kernel_regularizer='l2') for nodes, fn in dense_config]

    def call(self, inputs):
        x = self.embedding(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        return x

class ItemModel(tf.keras.Model):
    def __init__(self, embed_dim, dense_config, **kwargs):
        '''
            Dense Config is an array of tuples containing (nodes, activation)
        '''
        super().__init__(**kwargs)
        self.embedding = ItemEmbedding(10, embed_dim)
        self.dense_layers = [tf.keras.layers.Dense(nodes, activation=fn, kernel_regularizer='l2') for nodes, fn in dense_config]

    def call(self, inputs):
        x = self.embedding(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        return x

class TimeSeriesModel(tf.keras.Model):
    def __init__(self, timestamps, **kwargs):
        super().__init__(**kwargs)
        self.user = UserEmbedding(10, 8)
        self.item = ItemEmbedding(10, 8)
        self.time = tf.keras.layers.Normalization(axis=None)
        self.time.adapt((timestamps.astype(np.int64) // 10 ** 9).to_numpy())
        self.lstm_1 = tf.keras.layers.LSTM(1024) #, return_sequences=True)
        # self.lstm_2 = tf.keras.layers.LSTM(512)
        self.dense = tf.keras.layers.Dense(501, activation='sigmoid')
    
    def call(self, inputs):
        user = self.user(inputs)
        item = self.item(inputs)
        time = tf.reshape(self.time(inputs['t_dat']), (*inputs['t_dat'].shape, 1))
        features = tf.concat([user, item, time], axis=-1)
        outputs = self.lstm_1(features)
        # outputs = self.lstm_2(outputs)
        outputs = self.dense(outputs)
        return outputs

class TemporalModel(tf.keras.Model):
    def __init__(self, embed_dim, user_dense_config, item_dense_config, config, timestamps, **kwargs):
        '''
            Config is a dictionary having two keys lstm and dense
            Each key stores an array of the layer config as a tuple
            For lstm (nodes, return_sequences)
            For dense (nodes, activation)
        '''
        super().__init__(**kwargs)
        self.user = UserModel(embed_dim, user_dense_config)
        self.item = ItemModel(embed_dim, item_dense_config)
        self.time = tf.keras.layers.Normalization(axis=None)
        self.time.adapt(timestamps)

        self.lstm_layers = [tf.keras.layers.LSTM(nodes, return_sequences=ret_seq) for nodes, ret_seq in config['lstm']]
        self.dense_layers = [tf.keras.layers.Dense(nodes, activation=fn, kernel_regularizer='l2') for nodes, fn in config['dense']]

    def call(self, inputs):
        user = self.user(inputs)
        item = self.item(inputs)
        time = self.time(inputs['t_dat'])
        # time = tf.reshape(self.time(inputs['t_dat']), (*inputs['t_dat'].shape, 1))
        features = tf.concat([user, item, time], axis=-1)
        for layer in self.lstm_layers:
            features = layer(features)
        for layer in self.dense_layers:
            features = layer(features)
        return features