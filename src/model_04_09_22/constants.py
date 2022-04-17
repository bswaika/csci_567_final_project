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
    'pop_users': f'{DATA_DIR}/popular_customers.csv',
    'pop_items': f'{DATA_DIR}/popular_articles.csv',
    'users': f'{DATA_DIR}/customers.csv',
    'items': f'{DATA_DIR}/articles.csv',
    'transactions': f'{DATA_DIR}/transactions.csv',
    'master_intersection': f'{DATA_DIR}/master_AND.csv',
    'master_union': f'{DATA_DIR}/master_OR.csv'
}
DATA_SOURCE_DIR = './data/source'
DATA_SOURCE_TRACKER = './trackers/source_tracker.csv'
MODEL_TRACKER = './trackers/model_tracker.csv'
MODEL_DIR = './models'
DATETIME_FORMAT = '%Y-%m-%d'