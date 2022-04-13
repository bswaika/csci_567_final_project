import pandas as pd

gt_1yr_1wk = pd.read_csv('./predictions/gt_1yr_1wk.csv')
predictions_2yr_1wk = pd.read_csv('./predictions/predictions_2yr_1wk.csv')
train = pd.read_csv('../data/transactions_train.csv')

intersect = lambda x: x[x['customer_id'].isin(gt_1yr_1wk['customer_id'])]

train = intersect(train)
predictions_2yr_1wk = intersect(predictions_2yr_1wk)

train.to_csv('./predictions/train_subset.csv', index=False)
predictions_2yr_1wk.to_csv('./predictions/preds_2yr_1wk_subset.csv', index=False)

