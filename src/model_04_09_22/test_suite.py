from models import TimeSeriesModel, features
from constants import MODEL_DIR

import tensorflow as tf
from tqdm import tqdm
import numpy as np

MODEL_ID = 'model_a97ea1c6-3077-4cee-8fe2-fa1d32bc52b1_ff01743d-cb89-4ffe-8e88-0895218dea7f'

model = TimeSeriesModel(features['t_dat'])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adadelta(), metrics=[tf.keras.metrics.MeanSquaredError()])

model.load_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
print('model loaded...')

# predictions = model(X_test)
# BATCH_SIZE = 10
# predictions = []
# for i in tqdm(range(0, X_test['t_dat'].shape[0], BATCH_SIZE)):
#     x = {k: X_test[k][i:i+BATCH_SIZE] for k in X_test}
#     predictions.append(model(x))

# print(predictions[0])
# predictions = np.array(predictions).reshape(X_test['t_dat'].shape[0], len(Y_vocab))
# print(predictions[:10])


# print('writing predictions...')
# with open(f'{MODEL_DIR}/{MODEL_ID}/predictions.csv', 'w') as outfile:
#     for prediction in tqdm(predictions):
#         sorted_prediction = sorted(range(len(Y_vocab)), key=lambda x: prediction[x], reverse=True)
#         outfile.write(','.join(map(lambda x: str(Y_vocab[x]), sorted_prediction[:10])) + '\n')