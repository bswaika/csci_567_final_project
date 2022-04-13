'''
    Module
    ----------------------------------
    Implements functions used for handling 
    various IO functionality
'''

import os
import pandas as pd

def is_files_exist_in(dir, filenames_to_check):
    ''' Check if the necessary files exist in the specified directory

        Params
        ------
        dir : str
            the directory to check in
        filenames_to_check : list
            list of filenames to check in `dir`
        
        Returns
        -------
        bool
            whether the files exist or not
    '''
    files = os.listdir(dir)
    if all([f'{filename}.csv' in files for filename in filenames_to_check]):
        return True
    return False

def load_df(path, nrows=None) -> pd.DataFrame:
    ''' Load the Dataframe from the mentioned path

        Params
        ------
        path : str
            filepath to retrieve data from
        nrows : int, optional
            max number of rows to fetch data for
            used for debugging purposes to load the
            dataset fast (default None => load whole file)
        
        Returns
        -------
        pandas.DataFrame
            the dataframe loaded from the file
    '''
    return pd.read_csv(path, nrows=nrows)

def save_df(dataframe: pd.DataFrame, path):
    ''' Save the Dataframe to the mentioned path

        Params
        ------
        dataframe : pandas.DataFrame
            dataframe to write out
        path : str
            filepath to save data to
    '''
    dataframe.to_csv(path, index=False)

def select_columns(dataframe: pd.DataFrame, selectors) -> pd.DataFrame:
    ''' Select specified columns from dataframe and return the new dataframe

        Params
        ------
        dataframe : pandas.DataFrame
            dataframe to select columns from
        selectors : list
            List specifying the columns to select
            values from
        
        Returns
        -------
        pandas.DataFrame
            the dataframe with only the selected columns
    '''
    return dataframe[selectors]