from models import TemporalModel, UserModel, ItemModel
from models import get_trainable_data, get_encoded_labels, save_model, generate_predictions, save_predictions, train_test_split

import pandas as pd
import tensorflow as tf
import os, uuid

from constants import DATA_SOURCE_DIR, MODEL_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FEATURES_DIR = f'{DATA_SOURCE_DIR}/features'
LABELS_DIR = f'{DATA_SOURCE_DIR}/labels'
# FILE_ID = 'ff01743d-cb89-4ffe-8e88-0895218dea7f'
# FILE_ID = '7cb8b45d-d4b9-44d8-ba51-554ab92a0222'
# FILE_ID = '35c4c70c-9502-481e-a29a-e46a9e95260b'
FILE_ID = '603c644a-2350-47c1-a00d-de01e684db75'

features = pd.read_csv(f'{FEATURES_DIR}/{FILE_ID}.csv')
labels = pd.read_csv(f'{LABELS_DIR}/{FILE_ID}.csv')

X = get_trainable_data(features)
Y, label_vocab = get_encoded_labels(labels['labels'])

X = {k: X[k] if k != 'age' and k !='t_dat' else X[k].reshape(*X[k].shape, 1) for k in X}

# Param
SPLIT_RATIO = 0.8

train, test = train_test_split(X, Y, SPLIT_RATIO)
X_train, Y_train = train
X_test, Y_test = test

print(X_train['t_dat'].shape, Y_train.shape)
print(X_test['t_dat'].shape, Y_test.shape)

# HyperParams
EMBED_DIM = 8
USER_DENSE_CONFIG = [(7, 'relu')]
ITEM_DENSE_CONFIG = [(7, 'relu')]
CONFIG = {
    'lstm': [(128, True), (128, True), (128, False)],
    'dense': [(50, 'relu'), (501, 'sigmoid')]
}
TIMESTAMPS = X_train['t_dat']

TRAINING = True

model = TemporalModel(EMBED_DIM, USER_DENSE_CONFIG, ITEM_DENSE_CONFIG, CONFIG, TIMESTAMPS)


if TRAINING:
    # -----------------------
    # Config for 5 and 6
    # -----------------------
    # EMBED_DIM = 16
    # USER_DENSE_CONFIG = [(32, 'relu'), (16, 'relu'), (7, 'relu')]
    # ITEM_DENSE_CONFIG = [(32, 'relu'), (16, 'relu'), (7, 'relu')]
    # CONFIG = {
    #     'lstm': [(1024, True), (512, False)],
    #     'dense': [(512, 'relu'), (501, 'sigmoid')]
    # }
    # TIMESTAMPS = X_train['t_dat']

    # 5 - check above config
    # LEARNING_RATE = 0.1
    # EPOCHS = 1500
    # BATCH_SIZE = 50
    # model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer=tf.keras.optimizers.Adadelta(learning_rate=LEARNING_RATE), metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.TopKCategoricalAccuracy(10)])
    # model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    # loss, auc, topk = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    # MODEL_ID = f'model_{uuid.uuid4()}_{FILE_ID}'
    # os.mkdir(f'{MODEL_DIR}/{MODEL_ID}')
    # model.save_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
    # save_model(MODEL_ID, (loss, auc, topk))
    # predictions = generate_predictions(model, X_test, BATCH_SIZE, label_vocab)
    # save_predictions(predictions, MODEL_ID, label_vocab)

    # 6 - check above config
    # LEARNING_RATE = 0.1
    # EPOCHS = 250
    # BATCH_SIZE = 50
    # model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(), optimizer=tf.keras.optimizers.Adadelta(learning_rate=LEARNING_RATE), metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.TopKCategoricalAccuracy(10)])
    # model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    # loss, auc, topk = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    # MODEL_ID = f'model_{uuid.uuid4()}_{FILE_ID}'
    # os.mkdir(f'{MODEL_DIR}/{MODEL_ID}')
    # model.save_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
    # save_model(MODEL_ID, (loss, auc, topk))
    # predictions = generate_predictions(model, X_test, BATCH_SIZE, label_vocab)
    # save_predictions(predictions, MODEL_ID, label_vocab)

    # 7 - new config
    LEARNING_RATE = 0.1
    EPOCHS = 1500
    BATCH_SIZE = 50
    model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer=tf.keras.optimizers.Adadelta(learning_rate=LEARNING_RATE), metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.TopKCategoricalAccuracy(10)])
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    loss, auc, topk = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    MODEL_ID = f'model_{uuid.uuid4()}_{FILE_ID}'
    os.mkdir(f'{MODEL_DIR}/{MODEL_ID}')
    model.save_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
    save_model(MODEL_ID, (loss, auc, topk))
    predictions = generate_predictions(model, X_test, BATCH_SIZE, label_vocab)
    save_predictions(predictions, MODEL_ID, label_vocab)

    # Doesn't learn at all
    # LEARNING_RATE = 0.1
    # EPOCHS = 2000
    # BATCH_SIZE = 100
    # model.compile(loss=tf.keras.losses.KLDivergence(), optimizer=tf.keras.optimizers.Adadelta(learning_rate=LEARNING_RATE), metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.TopKCategoricalAccuracy(10)])
    # model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    # loss, auc, topk = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    # MODEL_ID = f'model_{uuid.uuid4()}_{FILE_ID}'
    # os.mkdir(f'{MODEL_DIR}/{MODEL_ID}')
    # model.save_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
    # save_model(MODEL_ID, (loss, auc, topk))
    # predictions = generate_predictions(model, X_test, BATCH_SIZE, label_vocab)
    # save_predictions(predictions, MODEL_ID, label_vocab)



# else:
#     MODEL_ID = 'model_a97ea1c6-3077-4cee-8fe2-fa1d32bc52b1_ff01743d-cb89-4ffe-8e88-0895218dea7f'
#     model.load_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
#     loss, mse = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)