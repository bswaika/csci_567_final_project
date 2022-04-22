'''
    Script
    ----------------------------------
    Prepares the master data in variety of time sequence
    data for feeding into RNNs. Note that labels are always
    for upto 20 items right after the training data 
    (could be more than a week)
    Variations:
        1. 1 month
        2. 1 week
        3. 3 months
        4. 6 months
    The training features are stored as rows of transactions
    associated with an index identifying the sequence
    The training labels are stored in a separate file

    Has to be called with flags 
        1. --weeks=w : Identifies the number of weeks to look at before a paricular week to generate train sequences
        2. --desc="Some description of the duration" : Describes the configuration in words to associate with the files for tracking
        3. --max=m : Maximum number of data points to generate
        4. --seq=s : Maximum sequence length to generate
        5. --shuffle : If this flag exists it selects `m` examples after shuffling, else takes `m` most recent
        6. --seasonal=y : If provided, tries to extract based on a seasonal pattern, i.e, takes w weeks before and also w weeks in the neighboorhood
                        of the current date from previous years. Here neighborhood means d +/- ((w * 7) // 2) from previous years.
                        Eg: If w=2 and point of interest is currently Mar 7, 20, then take Feb 21, 20 - Mar 7, 20 and also take
                        Mar 1, 19 - Mar 15, 19 and Mar 1, 18 - Mar 15, 18. `y` parameter controls the number of lookback years, but doesn't go
                        beyond whatever is available obviously, nor does it expect the user to know that. (Not Implemented YET)
        7. --master=OR|AND : If provided, uses master_master.csv file. The second master is the value of the param provided
'''

import sys, uuid, tqdm, hmac, hashlib
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from datetime import timedelta

from constants import DATA_SOURCE_DIR, DATA_FILES, DATA_SOURCE_TRACKER
from common import load_df, save_df
import transactions

def parse_flags(args):
    ''' Parse flags provided as a str array based on the specification
        mentioned above

        Params
        ------
        args : list[str]
            List of arguments to process. Must match the specification.
            No error handling has been done, thereby can crash, if it 
            doesn't match. Proceed with caution

        Returns
        -------
        config : dict{str : int | str | bool}
            Config dictionary that runs the script
    '''
    default_config = {
        'weeks': 4,
        'desc': '1 month prior to prediction',
        'max': 500_000,
        'seq': 250,
        'shuffle': False,
        'master': 'AND'
    }
    args = [str(arg).replace('--', '') for arg in args]
    default_config['shuffle'] = True if 'shuffle' in args else default_config['shuffle']
    if 'shuffle' in args:
        args.remove('shuffle')
    args = [arg.split('=') for arg in args]
    config = {k: v for k, v in args}
    for key in default_config:
        if not key in config:
            config[key] = default_config[key]
        else:
            config[key] = int(config[key]) if key != 'desc' and key != 'master' else str(config[key])
    config['id'] = str(uuid.uuid4())
    return config

def save_config(config):
    ''' Saves the config in the tracker file for tracking

        Params
        ------
        config : dict{str : int | str | bool}
            Config dictionary for the script
    '''
    with open(DATA_SOURCE_TRACKER, 'a') as outfile:
        seasonal = True if 'seasonal' in config else False
        years = config['seasonal'] if 'seasonal' in config else 0
        outfile.write(f'{config["id"]},{config["desc"]},{config["max"]},{config["seq"]},{config["shuffle"]},{seasonal},{years}\n')

def prepare_dataset(dataframe: pd.DataFrame, offset, num, seq_length) -> Tuple[list, pd.DataFrame, pd.DataFrame]:
    ''' Prepare dataset by pick points from the back of the dataframe
        and generating corresponding sequences

        Params
        ------
        dataframe : pandas.DataFrame
            the dataframe to extract points from
        num : int
            maximum number of points to generate
        shuffle : bool
            whether to shuffle or not
        seq_length : int
            maximum length of a sequence

        Returns
        -------
        points : list[(numpy.datetime64, str)]
            list of points generated
        features : pandas.DataFrame
            feature dataframe
        labels : pandas.DataFrame
            label dataframe
    '''
    feature_df = pd.DataFrame()
    label_df = {
        'id': [],
        'labels': []
    }
    df = dataframe.copy()
    start = dataframe['t_dat'].min()
    end = dataframe['t_dat'].max()
    dataframe = dataframe[dataframe['t_dat'] > (start + timedelta(days=offset*7))]
    dataframe = dataframe[dataframe['t_dat'] < (end - timedelta(days=7))]
    poi = set()
    i, j, dup, no_lab = 0, dataframe.shape[0] - 1, 0, 0
    pbar = tqdm.tqdm(total=num, desc='Generating Datapoints')
    while i < num and j >= 0:
        row = dataframe.loc[j, ['customer_id', 't_dat']]
        id, date = row['customer_id'], np.datetime64(row['t_dat'])
        if (date, id) not in poi:
            features, labels = generate_sequence(df, (date, id), offset, seq_length)
            if labels[1]:
                poi |= {(date, id)}
                feature_df = feature_df.append(features)
                label_df['id'].append(labels[0])
                label_df['labels'].append(labels[1])
                i += 1
                pbar.update(1)
            else:
                no_lab += 1
        else:
            dup += 1
        j -= 1
    pbar.close()
    print(f'Skipped {dup} duplicates')
    print(f'Skipped {no_lab} points without labels')
    print(f'Generated {len(poi)} points')
    return list(poi), feature_df.reset_index().drop(columns=['index']), pd.DataFrame(label_df)

def get_next_purchases(dataframe: pd.DataFrame, date, limit=20):
    ''' Get at most `limit` # next purchases from the given `date`
        for the user in the dataframe

        Params
        ------
        dataframe : pandas.DataFrame
            the dataframe to extract points from masked on a specific user
        date : numpy.datetime64
            date to start search from
        limit : int
            maximum length of a result

        Returns
        -------
        labels : list
            purchases after `date`
    '''
    time_mask = dataframe['t_dat'] > date
    labels = dataframe[time_mask].reset_index().drop(columns=['index'])['article_id'].to_list()
    return labels[:limit]


def generate_sequence(dataframe: pd.DataFrame, point, offset, seq_length):
    ''' Generates a sequence for a `point` of interest

        Params
        ------
        dataframe : pandas.DataFrame
            the dataframe to extract points from masked on a specific user
        point : tuple(numpy.datetime64, str)
            the user and the date for performing the transformation
        offset : int
            number of weeks to look in history

        Returns
        -------
        features : pandas.DataFrame
            dataframe with only user history
        labels : list
            purchases after the date query for the user
    '''
    date, id = point
    start = date - np.timedelta64(timedelta(days=offset*7))
    time_mask = (dataframe['t_dat'] >= start) & (dataframe['t_dat'] <= date)
    user_mask = dataframe['customer_id'] == id
    features = dataframe[time_mask & user_mask].reset_index().drop(columns=['index'])
    features = features.loc[features.shape[0]-seq_length:, :]
    uid = hmac.new(b'9196', (str(id) + str(date)).encode('utf-8'), hashlib.sha256).hexdigest()[:16]
    features['id'] = pd.Series(list(map(lambda x: f'{uid}_{x}', range(features.shape[0]))))

    labels = get_next_purchases(dataframe[user_mask], date)
    return (features, (uid, labels))

def split_data(user_id, user_transactions: pd.DataFrame, count, sequences, num_labels, sl_no) -> Optional[Tuple[str, pd.DataFrame, dict]]:
    purchase_densities = user_transactions.groupby('t_dat')['article_id'].count().reset_index().sort_values(by='t_dat')
    purchase_densities['article_cumfreq'] = purchase_densities['article_id'].cumsum() / count
    split_ratio = 0.75
    # Encourage the closest split to 75%
    lowest_diff_idx = abs(purchase_densities['article_cumfreq'] - split_ratio).argmin()
    split_point = purchase_densities.loc[lowest_diff_idx, 'article_cumfreq']
    # print(purchase_densities)
    # As long as it is not above 85%
    if split_point <= 0.85:
        split_at = purchase_densities['article_cumfreq'] <= split_point
    # Go back under 75%
    else:
        split_at = purchase_densities['article_cumfreq'] <= split_ratio
        # Get the actual split percentage
        if purchase_densities.loc[split_at, 'article_id'].sum() != 0:
            split_point = purchase_densities[split_at]['article_cumfreq'].iloc[-1]
        else:
            split_point = 0
    # If past is too little then ignore - since model would possibly not have 
    # enough temporal data to train on - emphasis on longer past
    if split_point < 0.35:
        return None
    else:
        seq_length = purchase_densities.loc[split_at, 'article_id'].sum()
        label_length = purchase_densities.loc[~split_at, 'article_id'].sum()
        split_date = purchase_densities.loc[purchase_densities['article_cumfreq'] == split_point, 't_dat'].reset_index().drop(columns='index')['t_dat'].values[0]
        time_mask = user_transactions['t_dat'] <= split_date
        features = user_transactions[time_mask].reset_index().drop(columns=['index']).loc[max(0, seq_length-sequences):, :]
        labels = user_transactions[~time_mask].reset_index().drop(columns=['index'])['article_id'].to_list()[:min(label_length, num_labels)]
        uid = hmac.new(b'9196', (str(user_id) + str(split_date)).encode('utf-8'), hashlib.sha256).hexdigest()[:16]
        uid += str(sl_no)
        features['id'] = uid
        return uid, features, labels

def prepare_datatset_v2(master: pd.DataFrame, sequences=45, num_labels=15):
    master['t_dat'] = master['t_dat'].view(dtype=np.int64) // 10 ** 9
    num_transactions_per_user = master.groupby('customer_id')['t_dat'].count().reset_index()
    num_transactions_per_user = num_transactions_per_user[num_transactions_per_user['t_dat'] >= 15]

    # print(num_transactions_per_user[(num_transactions_per_user['t_dat'] > 60) & (num_transactions_per_user['t_dat'] <= 100)].count())
    # print(num_transactions_per_user[(num_transactions_per_user['t_dat'] > 100)])

    # Dont accomodate too many purchases - can skew the model - a pattern exists where these people
    # are buying 100+ items in one day
    condition = num_transactions_per_user['t_dat'] <= 120 
    num_transactions_per_user = num_transactions_per_user[condition].reset_index().drop(columns=['index'])
    num_taken = 0

    feature_df = pd.DataFrame()
    label_df = {
        'id': [],
        'labels': []
    }

    for user_id, count in tqdm.tqdm(num_transactions_per_user.values):
        mask = master['customer_id'] == user_id
        user_transactions = master[mask].reset_index().drop(columns=['index'])
        if count < 65:
            # Generate a single data point - ~8949
            result = split_data(user_id, user_transactions, count, sequences, num_labels, num_taken)
            if result:
                uid, features, labels = result
                feature_df = feature_df.append(features)
                label_df['id'].append(uid)
                label_df['labels'].append(labels)
                num_taken += 1
        else:
            # Try to generate multiple data points - ~86
            BATCH_SIZE = 60
            # Slide window of 60 transactions over 5 transactions
            INCREMENTS = 5
            i = 0
            while i + BATCH_SIZE <= count:
                result = split_data(user_id, user_transactions.loc[i:i+BATCH_SIZE-1, :], BATCH_SIZE, sequences, num_labels, num_taken)
                if result:
                    uid, features, labels = result
                    feature_df = feature_df.append(features)
                    label_df['id'].append(uid)
                    label_df['labels'].append(labels)
                    num_taken += 1
                i += INCREMENTS   
    return feature_df.reset_index().drop(columns='index'), pd.DataFrame(label_df)

def load():
    # Generate config from the command line flags
    config = parse_flags(sys.argv[1:])

    IN_FILE = DATA_FILES['master_intersection'] if config['master'] == 'AND' else DATA_FILES['master_union']
    
    # Load master dataframe and sort by date
    master = load_df(IN_FILE)
    master = transactions.set_date_column_type(master)
    master = master.sort_values(by='t_dat').reset_index().drop(columns=['index'])

    return config, master

def save(features, labels, config, OUT_FILES):
    # Save dataset
    save_df(features, OUT_FILES['features'])
    save_df(labels, OUT_FILES['labels'])

    # Save config
    save_config(config)

if __name__ == "__main__":
    config, master = load()
    OUT_FILES = {
        'features': f'{DATA_SOURCE_DIR}/features/{config["id"]}.csv',
        'labels': f'{DATA_SOURCE_DIR}/labels/{config["id"]}.csv'        
    }

    # Prepare dataset
    # points, features, labels = prepare_dataset(master, config['weeks'], config['max'], config['seq'])

    # Prepare dataset
    features, labels = prepare_datatset_v2(master)
    print(features.shape, labels.shape)

    save(features, labels, config, OUT_FILES)
    