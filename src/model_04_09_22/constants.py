'''
    Module
    ----------------------------------
    Stores constants required across 
    other modules and scripts
'''
NUM_USERS = 200_000
NUM_ITEMS = 500
TRAIN_DATA_DURATION_DAYS = 365
TEST_DATA_DURATION_DAYS = 7
RAW_DATA_DIR = '../../data'
RAW_DATA_FILES = {
    'users': f'{RAW_DATA_DIR}/customers.csv',
    'items': f'{RAW_DATA_DIR}/articles.csv',
    'transactions': f'{RAW_DATA_DIR}/transactions_train.csv'
}
DATA_DIR = './data'
DATA_FILES = {
    'users': f'{DATA_DIR}/customers.csv',
    'items': f'{DATA_DIR}/articles.csv',
    'transactions': f'{DATA_DIR}/transactions.csv',
    'master': f'{DATA_DIR}/master.csv'
}
MODEL_DIR = './models'
DATETIME_FORMAT = '%Y-%m-%d'