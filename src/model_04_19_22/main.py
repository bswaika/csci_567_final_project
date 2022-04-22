import os, time
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from lightgbm import Dataset, train as train_booster
from constants import SAMPLE_DIR, PREDICTION_DIR, PREDICTION_FILE, GROUP_FILE, MODEL_DIR, MODEL_FILE
from load import load_users

def train_test_split(data: pd.DataFrame, split_ratio):
    counter = data.groupby(['customer_id', 'timestamp'], sort=False)['article_id'].count().cumsum()
    counter = counter / data['article_id'].count()
    lowest_diff_idx = abs(counter - split_ratio).argmin()
    counter = counter.reset_index()
    train, test = counter.loc[:lowest_diff_idx, ['customer_id', 'timestamp']], counter.loc[lowest_diff_idx+1:, ['customer_id', 'timestamp']]
    train_mask = data['customer_id'].isin(train['customer_id']) & data['timestamp'].isin(train['timestamp'])
    test_mask = data['customer_id'].isin(test['customer_id']) & data['timestamp'].isin(test['timestamp'])
    return data[train_mask].reset_index().drop(columns=['index']), data[test_mask].reset_index().drop(columns=['index'])


def load_strategy_BNM() -> Tuple[pd.DataFrame, str]:
    STRATEGY_STRING = 'BNM'
    files = os.listdir(SAMPLE_DIR)
    bnm_files = filter(lambda file: file.split('_')[-1].split('.')[0] == STRATEGY_STRING, files)

    sample_data = pd.DataFrame()
    for file in bnm_files:
        sample = pd.read_csv(f'{SAMPLE_DIR}/{file}')
        sample_data = sample_data.append(sample)
        sample_data = sample_data.drop_duplicates(ignore_index=True)
    
    return sample_data.reset_index().drop(columns=['index']), STRATEGY_STRING

def load_strategy_split_BNM(test_file_id) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    STRATEGY_STRING = 'BNM'
    files = os.listdir(SAMPLE_DIR)
    bnm_files = filter(lambda file: file.split('_')[-1].split('.')[0] == STRATEGY_STRING, files)

    sample_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for file in bnm_files:
        if not test_file_id in file:
            sample = pd.read_csv(f'{SAMPLE_DIR}/{file}')
            sample_data = sample_data.append(sample)
            sample_data = sample_data.drop_duplicates(ignore_index=True)
            print(f'loaded {file} for train...')
        else:
            test_data = pd.read_csv(f'{SAMPLE_DIR}/{file}')
            print(f'loaded {file} for test...')
    
    return sample_data.reset_index().drop(columns=['index']), test_data, STRATEGY_STRING


users = load_users()
# data, STRATEGY_STRING = load_strategy_BNM()
# Construct categorical features - Note that article_id and customer_id are included since lgbm only allows int float or bool
# categorical_feature_cols = [col for col in data.columns if 'name' in col] + ['FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'postal_code', 'sales_channel_id']
# for col in categorical_feature_cols:
#     data[col] = data[col].astype('category')
# train, test = train_test_split(data, 0.8)


train, test, STRATEGY_STRING = load_strategy_split_BNM('4911218')
# Construct categorical features - Note that article_id and customer_id are included since lgbm only allows int float or bool
categorical_feature_cols = [col for col in train.columns if 'name' in col] + ['FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'postal_code', 'sales_channel_id']
# with open('./data/train_categories.txt', 'w') as outfile:
for col in categorical_feature_cols:
    train[col] = train[col].astype('category')
    test[col] = pd.Series(pd.Categorical(test[col], categories=train[col].cat.categories))
        # outfile.write(f'{col}:{",".join(map(str, list(train[col].cat.categories)))}\n')

train = train.sort_values(by=['customer_id', 'timestamp']).reset_index().drop(columns='index')
test = test.sort_values(by=['customer_id', 'timestamp']).reset_index().drop(columns='index')

# Normalize continuous features
scalers = {
    'age': StandardScaler().fit(train['age'].to_numpy().reshape(-1, 1)),
    'timestamp': StandardScaler().fit(train['timestamp'].to_numpy().reshape(-1, 1)),
    'avg_price': StandardScaler().fit(train['avg_price'].to_numpy().reshape(-1, 1)),
    'popularity': StandardScaler().fit(train['popularity'].to_numpy().reshape(-1, 1))
}

# with open('./data/train_scalers.txt', 'w') as outfile:
#     for key in scalers:
#         outfile.write(f'{key}:{scalers[key].mean_[0]},{scalers[key].scale_[0]}\n')

for key in scalers:
    train[key] = scalers[key].transform(train[key].to_numpy().reshape(-1, 1)).flatten()
    test[key] = scalers[key].transform(test[key].to_numpy().reshape(-1, 1)).flatten()

# Encode Ordinal features - customer_id
encoder = OrdinalEncoder().fit(users['customer_id'].to_numpy().reshape(-1, 1))
# with open('./data/train_encoder_customer_id.txt', 'w') as outfile:
#     outfile.write(f'customer_id:{",".join(map(str, list(encoder.categories_[0])))}\n')
train['customer_id'] = encoder.transform(train['customer_id'].to_numpy().reshape(-1, 1)).flatten()
test['customer_id'] = encoder.transform(test['customer_id'].to_numpy().reshape(-1, 1)).flatten()

X_train, Y_train = train.drop(columns=['relevance', 'sales_channel_id']), train['relevance']
X_test, Y_test = test.drop(columns=['relevance', 'sales_channel_id']), test['relevance']

group_train = X_train.groupby('customer_id')['timestamp'].count().to_numpy()
group_test = X_test.groupby('customer_id')['timestamp'].count().to_numpy()

train_ds = Dataset(X_train, Y_train, group=group_train).construct()
test_ds = Dataset(X_test, Y_test, group=group_test, free_raw_data=False).construct()

params = {
    'num_leaves': 32,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'objective': 'lambdarank',
    'reg_lambda': 0.05,
    'n_jobs': 4,
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10, 15],
    'early_stopping_rounds': 5
}

model = train_booster(
    params=params,
    train_set=train_ds,
    valid_sets=[test_ds],
    valid_names=['test_set'],
    verbose_eval=1
)

ts = int(time.time())
model.save_model(MODEL_FILE.format(dir=MODEL_DIR, timestamp=ts, strategy=STRATEGY_STRING))

predictions = model.predict(test_ds.get_data())
predictions = pd.DataFrame({
    'predictions': predictions,
    'labels': Y_test.to_numpy()
})

predictions.to_csv(PREDICTION_FILE.format(dir=PREDICTION_DIR, timestamp=ts, strategy=STRATEGY_STRING), index=False)
with open(GROUP_FILE.format(dir=PREDICTION_DIR, timestamp=ts, strategy=STRATEGY_STRING), 'w') as outfile:
    outfile.write('\n'.join(map(str, group_test)))