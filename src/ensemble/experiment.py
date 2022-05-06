import pandas as pd

kaggle = pd.read_csv('./submissions/submission_7.csv').sort_values('customer_id').reset_index(drop=True)
prepped = pd.read_csv('./submissions/sub_0237.csv').sort_values('customer_id').reset_index(drop=True)

kaggle['prediction'] = kaggle['prediction'].apply(lambda s: set(s.split()))
prepped['prediction'] = prepped['prediction'].apply(lambda s: set(s.split()))

# print(kaggle['prediction'].head())
# print(prepped['prediction'].head())

df = prepped
# df.columns = ['customer_id', 'prepped']
df['kaggle'] = kaggle['prediction']

del kaggle

df['int_count'] = df.apply(lambda x: len(x['prediction'] & x['kaggle']), axis=1)
print(df['int_count'].mean())

# def generate_preds(x, kaggle=True):
#     preds = list(x['prepped'] & x['kaggle'])
#     if len(preds) < 12:
#         if kaggle:
#             preds += list(x['kaggle'] - x['prepped'])
#         else:
#             preds += list(x['prepped'] - x['kaggle'])
#     return ' '.join(preds)


# df['prediction_k'] = df.apply(lambda x: generate_preds(x), axis=1)
# df['prediction_p'] = df.apply(lambda x: generate_preds(x, False), axis=1)

# sub1 = df[['customer_id', 'prediction_k']]
# sub1.columns = ['customer_id', 'prediction']

# sub2 = df[['customer_id', 'prediction_p']]
# sub2.columns = ['customer_id', 'prediction']

# # sub1.to_csv('./submissions/submission_2.csv', index=False)
# # sub2.to_csv('./submissions/submission_3.csv', index=False)

# print(sub1.tail())