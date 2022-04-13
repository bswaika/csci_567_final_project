'''
    Module
    ----------------------------------
    Implements functions used for handling 
    item data
'''

import pandas as pd

def fill(dataframe: pd.DataFrame) -> pd.DataFrame:
    ''' Fills NA values in given users dataframe for columns `age`, and `club_member_status`

        Params
        ------
        dataframe : pandas.DataFrame
            input dataframe to fill
        
        Returns
        -------
        dataframe : pandas.DataFrame
            output dataframe which has been filled
    '''
    dataframe['age'].fillna(0, inplace=True)
    dataframe['club_member_status'].fillna('NONE', inplace=True)
    return dataframe