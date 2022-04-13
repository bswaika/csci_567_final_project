import pandas as pd

class DataParser:
  def __init__(self, filepaths):
    assert isinstance(filepaths, dict)
    assert 'articles' in filepaths
    assert 'customers' in filepaths
    assert 'transactions' in filepaths
    self._filepaths = filepaths
    self._raw = {}
    # self._associations = {}
    self._data = {}
    self._encoders = {
      'articles': self.encode_articles,
      'customers': self.encode_customers,
      'transactions': self.encode_transactions,
    }
  
  def load_data(self, index):
    assert index in ['articles', 'customers', 'transactions']
    self._raw[index] = pd.read_csv(self._filepaths[index])
  
  def encode_data(self, index):
    assert index in ['articles', 'customers', 'transactions']
    assert index in self._raw
    self._encoders[index]()

  def create_association(self, index, id, name):  
    association_dict = self._raw[index][[id, name]]
    association_dict = association_dict.set_index([id]).to_dict()[name]
    return pd.DataFrame.from_dict(association_dict, orient='index', columns=['name'])
  
  def encode_articles(self):
    selectors = [col for col in self._raw['articles'].columns if any(['id' in col, 'code' in col, 'no' in col])]
    self._data['articles'] = self._raw['articles'].loc[:, selectors].copy(deep=True)

    # self._associations['product_types'] = self.create_association('articles', 'product_type_no', 'product_type_name')
    # self._associations['departments'] = self.create_association('articles', 'department_no', 'department_name')
    # self._associations['index_groups'] = self.create_association('articles', 'index_group_no', 'index_group_name')
    # self._associations['sections'] = self.create_association('articles', 'section_no', 'section_name')
    # self._associations['garment_groups'] = self.create_association('articles', 'garment_group_no', 'garment_group_name')
    # self._associations['graphical_appearances'] = self.create_association('articles', 'graphical_appearance_no', 'graphical_appearance_name')
    # self._associations['colour_groups'] = self.create_association('articles', 'colour_group_code', 'colour_group_name')
    # self._associations['indexes'] = self.create_association('articles', 'index_code', 'index_name')
    # self._associations['perceived_colour_values'] = self.create_association('articles', 'perceived_colour_value_id', 'perceived_colour_value_name')
    # self._associations['perceived_colour_masters'] = self.create_association('articles', 'perceived_colour_master_id', 'perceived_colour_master_name')

    self._data['articles']['index_code'] = self._data['articles']['index_code'].astype('category')
    # self._associations['index_codes'] = self._data['articles']['index_code'].cat.categories
    self._data['articles']['index_code'] = self._data['articles']['index_code'].cat.codes
  
  def fill_customers(self):
    self._raw['customers']['age'].fillna(0.0, inplace=True)
    self._raw['customers']['club_member_status'].fillna('NONE', inplace=True)
    self._raw['customers']['Active'].fillna(0.0, inplace=True)
    self._raw['customers']['FN'].fillna(0.0, inplace=True)
    self._raw['customers']['fashion_news_frequency'].fillna('NONE', inplace=True)
    self._raw['customers']['fashion_news_frequency'] = self._raw['customers']['fashion_news_frequency'].str.replace('NONE', 'None')
  
  def encode_customers(self):
    self.fill_customers()
    self._data['customers'] = self._raw['customers'].copy(deep=True)

    self._data['customers']['fashion_news_frequency'] = self._data['customers']['fashion_news_frequency'].astype('category')
    self._data['customers']['club_member_status'] = self._data['customers']['club_member_status'].astype('category')
    self._data['customers']['Active'] = self._data['customers']['Active'].astype('category')
    self._data['customers']['FN'] = self._data['customers']['FN'].astype('category')

    # self._associations['member_statuses'] = self._data['customers']['club_member_status'].cat.categories
    self._data['customers']['club_member_status'] = self._data['customers']['club_member_status'].cat.codes

    # self._associations['news_freqs'] = self._data['customers']['fashion_news_frequency'].cat.categories
    self._data['customers']['fashion_news_frequency'] = self._data['customers']['fashion_news_frequency'].cat.codes
  
  def encode_transactions(self):
    self._data['transactions'] = self._raw['transactions'].copy(deep=True)

  def get_data(self, index) -> pd.DataFrame:
    assert index in ['articles', 'customers', 'transactions']
    if index not in self._raw and index not in self._data:
      self.load_data(index)
      print(f'{index.upper()} loaded...')
    if index not in self._data:
      self.encode_data(index)
      del self._raw[index]
    return self._data[index]
  
  # def get_association(self, index) -> pd.DataFrame:
  #   assert index in self._associations
  #   return self._associations[index]