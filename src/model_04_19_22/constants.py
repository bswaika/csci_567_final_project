RAW_DATA_DIR = '../../data'
RAW_DATA_FILES = {
    'users': f'{RAW_DATA_DIR}/customers.csv',
    'items': f'{RAW_DATA_DIR}/articles.csv',
    'transactions': f'{RAW_DATA_DIR}/transactions_train.csv'
}
DATA_DIR = './data'
DATA_FILES = {
    'holdout': '{dir}/transactions_holdout_{weeks}.csv'
}
SAMPLE_DIR = './samples'
SAMPLE_FILE = '{dir}/sample_{id}_{strategy}.csv'
PREDICTION_DIR = './predictions'
PREDICTION_FILE = '{dir}/prediction_{timestamp}_{strategy}.csv'
GROUP_FILE = '{dir}/group_{timestamp}_{strategy}.txt'
MODEL_DIR = './models'
MODEL_FILE = '{dir}/model_{timestamp}_{strategy}.txt'
DATETIME_FORMAT = '%Y-%m-%d'