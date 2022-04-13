'''
    Module
    ----------------------------------
    Implements functions used for handling 
    transactions data
'''

import pandas as pd
from datetime import timedelta

from constants import DATETIME_FORMAT, TRAIN_DATA_DURATION_DAYS

def set_date_column_type(dataframe: pd.DataFrame) -> pd.DataFrame:
    ''' Sets the column type of `t_dat` column in the dataframe as datetime

        Params
        ------
        dataframe : pandas.DataFrame
            input dataframe to conert `t_dat` column to datetime
        
        Returns
        -------
        dataframe : pandas.DataFrame
            output dataframe having the converted column type
    '''
    dataframe['t_dat'] = pd.to_datetime(dataframe['t_dat'], format=DATETIME_FORMAT)
    return dataframe

def extract_subset(dataframe: pd.DataFrame, duration=TRAIN_DATA_DURATION_DAYS, from_start=True) -> pd.DataFrame:
    ''' Extracts data from the dataframe based on the given duration and unit

        Params
        ------
        dataframe : pandas.DataFrame
            input dataframe to extract data from
        duration : int, optional
            duration (in days) for which data needs to be extracted (default constants.TRAIN_DATA_DURATION_YEARS)
        from_start : bool, optional
            whether to extract from start or from end (default True)
        
        Returns
        -------
        dataframe : pandas.DataFrame
            output dataframe having the converted column type
    '''
    if from_start:
        start = dataframe['t_dat'].min()
        end = start + timedelta(days=duration)
    else:
        end = dataframe['t_dat'].max()
        start = end - timedelta(days=duration)

    mask = (dataframe['t_dat'] >= start) & (dataframe['t_dat'] <= end)
    return dataframe[mask].reset_index()

def compute_most_popular(dataframe: pd.DataFrame, index, num_to_retrieve) -> pd.DataFrame:
    ''' Computes most popular indices for the index specified from the dataframe

        Params
        ------
        dataframe : pandas.DataFrame
            input dataframe to compute popularity from
        index : str
            the index to compute popularity on
        num_to_retrieve : int
            number of items to retrieve for the mentioned index after computing popularity
        
        Returns
        -------
        series : pandas.DataFrame
            output series sorted in order of popularity with a length of `num_to_retrieve` with columns `index`, count
    '''
    return dataframe[index].value_counts().reset_index().rename(columns={'index': index, index: 'count'}).loc[:num_to_retrieve-1, :]

def extract_subset_union(dataframe: pd.DataFrame, users, items) -> pd.DataFrame:
    ''' Extracts data from the dataframe based on the union of users list and items list

        Params
        ------
        dataframe : pandas.DataFrame
            input dataframe to extract data from
        users : list
            list of users to condition upon
        items : list
            list of items to condition upon
        
        Returns
        -------
        dataframe : pandas.DataFrame
            output dataframe having rows with either customer_id in `users` list or article_id in `items` list or both
    '''
    return dataframe[dataframe['customer_id'].isin(users) | dataframe['article_id'].isin(items)]

def extract_subset_intersection(dataframe: pd.DataFrame, users, items) -> pd.DataFrame:
    ''' Extracts data from the dataframe based on the union of users list and items list

        Params
        ------
        dataframe : pandas.DataFrame
            input dataframe to extract data from
        users : list
            list of users to condition upon
        items : list
            list of items to condition upon
        
        Returns
        -------
        dataframe : pandas.DataFrame
            output dataframe having rows with customer_id in `users` list and article_id in `items` list
    '''
    return dataframe[dataframe['customer_id'].isin(users) & dataframe['article_id'].isin(items)]

def compute_targets(dataframe: pd.DataFrame) -> pd.DataFrame:
    ''' Aggregates transactions data to compute targets `freq` and `spend_freq`

        Params
        ------
        dataframe : pandas.DataFrame
            input dataframe to process aggregation on
        
        Returns
        -------
        dataframe : pandas.DataFrame
            output dataframe with target columns `freq` and `spend_freq`
    '''
    dataframe = dataframe.groupby(['customer_id', 'article_id'])[['t_dat', 'price']].agg({'t_dat': 'count', 'price': 'mean'}).reset_index().rename(columns={'t_dat': 'qty', 'price': 'avg_price'})
    dataframe['spend'] = dataframe['qty'] * dataframe['avg_price']
    totals = dataframe.groupby('customer_id')[['qty', 'spend']].sum().reset_index().rename(columns={'qty': 'total_qty', 'spend': 'total_spend'})
    dataframe = dataframe.merge(totals, on='customer_id')
    dataframe['spend_freq'] = dataframe['spend'] / dataframe['total_spend']
    dataframe['freq'] = dataframe['qty'] / dataframe['total_qty']

    return dataframe