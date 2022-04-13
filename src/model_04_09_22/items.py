'''
    Module
    ----------------------------------
    Implements functions used for handling 
    item data
'''

import pandas as pd

def generate_keywords(dataframe: pd.DataFrame) -> pd.DataFrame:
    ''' Generate keywords column for the input items dataframe

        Params
        ------
        dataframe : pandas.DataFrame
            input dataframe to generate keywords from
        
        Returns
        -------
        dataframe : pandas.DataFrame
            output dataframe having the keywords column
    '''
    keyword_cols = [c for c in dataframe.columns if c != 'prod_name' and c != 'article_id']
    
    def get_keyword_string(article):
        special_chars = {'&', 'and', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}
        transform_lowercase = lambda x: x.lower()
        remove_slashes = lambda x: ' '.join(x.split('/')) if '/' in x else x
        remove_comma = lambda x: x.replace(',', '')
        remove_plus = lambda x: x.replace('+', '')
        remove_minus = lambda x: x.replace('-', '')
        keyword_list = ' '.join(
                                map(remove_minus,
                                    map(remove_plus, 
                                        map(remove_comma, 
                                            map(remove_slashes, 
                                                map(transform_lowercase, 
                                                        article[keyword_cols])
                                                )
                                            )
                                        )
                                    )
                                ).split(' ')
        unique_keywords = list(set(keyword_list) - special_chars)
        return ' '.join(unique_keywords)
    
    dataframe['keywords'] = dataframe.apply(get_keyword_string, axis=1)
    return dataframe