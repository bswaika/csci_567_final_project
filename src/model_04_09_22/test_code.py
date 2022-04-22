# Test on a subset
# test_features = features.loc[:100, :]
# test_features['t_dat'] = test_features['t_dat'].astype(np.int64) // 10 ** 9
# test_features['id'] = test_features['id'].apply(lambda x: x[:16])
# max_len = test_features.groupby('id')['customer_id'].count().reset_index()['customer_id'].max()
# test_features = test_features.groupby('id').agg(lambda x: list(x)).reset_index().drop(columns='id')
# FILE_ID = '7cb8b45d-d4b9-44d8-ba51-554ab92a0222'

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

# X = get_trainable_data(features)
# gc.collect()
# Y, Y_vocab = get_encoded_labels(labels['labels'])
# gc.collect()

# # print(sample_weights.shape)
# print(X['t_dat'].shape, Y.shape)



# sample_weight_train = sample_weights[:train_size]

# Reconciliation checks for seeing if certain ids don't exist or stuff like that
# print(len(features['id'].unique()), len(labels['id'].unique()))
# print(features.loc[~features['id'].isin(labels['id']), 'id'])
# lbl_grouper = labels.groupby('id')['labels'].count().reset_index()
# print(lbl_grouper[lbl_grouper['labels'] > 1])
# print(lbl_grouper[lbl_grouper['labels'] > 1].count())


# -------------------------
# Some models trained earlier
# -------------------------

# EPOCHS = 25
# BATCH_SIZE = 50

# # 3 - SoftMax + BC - 25 - LSTM(1024) LSTM(512)
# model = TimeSeriesModel(features['t_dat'])
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adadelta(), metrics=[tf.keras.metrics.MeanSquaredError()])
# model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
# loss, mse = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
# MODEL_ID = f'model_{uuid.uuid4()}_{FILE_ID}'
# os.mkdir(f'{MODEL_DIR}/{MODEL_ID}')
# model.save_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
# save_model(MODEL_ID, loss)
# predictions = model(X_test)
# print(loss, mse)

# with open(f'{MODEL_DIR}/{MODEL_ID}/predictions.csv', 'w') as outfile:
#     for prediction in predictions:
#         sorted_prediction = sorted(range(len(Y_vocab)), key=lambda x: prediction[x], reverse=True)
#         outfile.write(','.join(map(lambda x: str(Y_vocab[x]), sorted_prediction[:10])) + '\n')

# EPOCHS = 30
# BATCH_SIZE = 50

# 4 - Sigmoid + BC - 30 - LSTM(1024)
# model = TimeSeriesModel(features['t_dat'])
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adadelta(), metrics=[tf.keras.metrics.MeanSquaredError()])
# model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
# loss, mse = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
# MODEL_ID = f'model_{uuid.uuid4()}_{FILE_ID}'
# os.mkdir(f'{MODEL_DIR}/{MODEL_ID}')
# model.save_weights(f'{MODEL_DIR}/{MODEL_ID}/checkpoint')
# save_model(MODEL_ID, loss)
# predictions = model(X_test)
# print(loss, mse)

# with open(f'{MODEL_DIR}/{MODEL_ID}/predictions.csv', 'w') as outfile:
#     for prediction in predictions:
#         sorted_prediction = sorted(range(len(Y_vocab)), key=lambda x: prediction[x], reverse=True)
#         outfile.write(','.join(map(lambda x: str(Y_vocab[x]), sorted_prediction[:10])) + '\n')