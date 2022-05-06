import numpy as np, pandas as pd, datetime as dt
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
from collections import defaultdict

def iter_to_str(iterable):
    return " ".join(map(lambda x: str(0) + str(x), iterable))

def apk(actual, predicted, k=12):
    if len(predicted) > k:
        predicted = predicted[:k]
    score, nhits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            nhits += 1.0
            score += nhits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=12, return_apks=False):
    assert len(actual) == len(predicted)
    apks = [apk(ac, pr, k) for ac, pr in zip(actual, predicted) if 0 < len(ac)]
    if return_apks:
        return apks
    return np.mean(apks)

def blend(dt, w=[], k=12):
    if len(w) == 0:
        w = [1] * (len(dt))
    preds = []
    for i in range(len(w)):
        preds.append(dt[i].split())
    res = {}
    for i in range(len(preds)):
        if w[i] < 0:
            continue
        for n, v in enumerate(preds[i]):
            if v in res:
                res[v] += (w[i] / (n + 1))
            else:
                res[v] = (w[i] / (n + 1))    
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return ' '.join(res[:k])

def prune(pred, ok_set, k=12):
    pred = pred.split()
    post = []
    for item in pred:
        if int(item) in ok_set and not item in post:
            post.append(item)
    return " ".join(post[:k])

def validation(actual, predicted, grouping, score=0, index=-1, ignore=False, figsize=(12, 6)):
    # actual, predicted : list of lists
    # group : pandas Series
    # score : pandas DataFrame
    if ignore: return
    ap12 = mapk(actual, predicted, return_apks=True)
    map12 = round(np.mean(ap12), 6)
    if isinstance(score, int): score = pd.DataFrame({g:[] for g in sorted(grouping.unique().tolist())})
    if index == -1 : index = score.shape[0]
    score.loc[index, "All"] = map12
    # plt.figure(figsize=figsize)
    # plt.subplot(1, 2, 1); sns.histplot(data=ap12, log_scale=(0, 10), bins=20); plt.title(f"MAP@12 : {map12}")
    for g in grouping.unique():
        map12 = round(mapk(actual[grouping == g], predicted[grouping == g]), 6)
        score.loc[index, g] = map12
    # plt.subplot(1, 2, 2); score[[g for g in grouping.unique()[::-1]] + ['All']].loc[index].plot.barh(); plt.title(f"MAP@12 of Groups")
    vc = pd.Series(predicted).apply(len).value_counts()
    score.loc[index, "Fill"] = round(1 - sum(vc[k] * (12 - k) / 12 for k in (set(range(12)) & set(vc.index))) / len(actual), 3) * 100
    print(score)
    return score

df = pd.read_parquet('./data/transactions_train.parquet')
sub = pd.read_csv('../../data/sample_submission.csv')
cid = pd.DataFrame(sub.customer_id.apply(lambda s: int(s[-16:], 16)))

group = df.groupby('customer_id').sales_channel_id.mean().round().reset_index().merge(cid, on='customer_id', how='right').rename(columns={'sales_channel_id':'group'})
grouping = group.group.fillna(1.0)

val_week = 105
# id of week to be used in a validation; set 105 if you would like to create a submission
val = df.loc[df.week == val_week].groupby('customer_id').article_id.apply(iter_to_str).reset_index().merge(cid, on='customer_id', how='right')
actual = val.article_id.apply(lambda s: [] if pd.isna(s) else s.split())
last_date = df.loc[df.week < val_week].t_dat.max()

init_date = last_date - dt.timedelta(days=9999)
train = df.loc[(df.t_dat >= init_date) & (df.t_dat <= last_date)].copy()
train = train.merge(train.groupby('customer_id').t_dat.max().reset_index().rename(columns={'t_dat':'l_dat'}),
                   on = 'customer_id', how='left')
train['d_dat'] = (train.l_dat - train.t_dat).dt.days
train = train.loc[train.d_dat < 14].sort_values(['t_dat'], ascending=False).drop_duplicates(['customer_id', 'article_id'])
sub['last_purchase'] = train.groupby('customer_id')\
    .article_id.apply(iter_to_str).reset_index()\
    .merge(cid, on='customer_id', how='right').article_id.fillna('')

predicted = sub['last_purchase'].apply(lambda s: [] if pd.isna(s) else s.split())
score = validation(actual, predicted, grouping, index='Last Purchase', ignore=(val_week == 105))

init_date = last_date - dt.timedelta(days=6)
train = df.loc[(df.t_dat >= init_date) & (df.t_dat <= last_date)].copy()\
    .groupby(['article_id']).t_dat.count().reset_index()
adf = pd.read_parquet('./data/articles.parquet')
adf = adf.merge(train, on='article_id', how='left').rename(columns={'t_dat':'ct'})\
    .sort_values('ct', ascending=False).query('ct > 0')

map_to_col = defaultdict(list)
for aid in adf.article_id.tolist():
    map_to_col[aid] = list(filter(lambda x: x != aid, adf[adf.product_code == aid // 1000].article_id.tolist()))[:1]

def map_to_variation(s):
    f = lambda item: iter_to_str(map_to_col[int(item)])
    return ' '.join(map(f, s.split()))
sub['other_colors'] = sub['last_purchase'].fillna('').apply(map_to_variation)

predicted = sub['other_colors'].apply(lambda s: [] if pd.isna(s) else s.split())
score = validation(actual, predicted, grouping, score, index='Other Colors', ignore=(val_week == 105))

init_date = last_date - dt.timedelta(days=5 - 1)
group_df = pd.concat([cid, group.group.fillna(1)], axis=1) # grouping can be changed
group_df.columns = ['customer_id', 'group']
train = df.loc[(df.t_dat >= init_date) & (df.t_dat <= last_date)].copy()\
    .merge(group_df, on='customer_id', how='left')\
    .groupby(['group', 'article_id']).t_dat.count().reset_index()
items = defaultdict(str)
for g in train.group.unique():
    items[g] = iter_to_str(train.loc[train.group == g].sort_values('t_dat', ascending=False).article_id.tolist()[:12])

sub['popular_items'] = group_df.group.map(items)

predicted = sub['popular_items'].apply(lambda s: [] if pd.isna(s) else s.split())
score = validation(actual, predicted, grouping, score, index='Popular Items', ignore=(val_week == 105))

init_date = last_date - dt.timedelta(days=11)
sold_set = set(df.loc[(df.t_dat >= init_date) & (df.t_dat <= last_date)].article_id.tolist())
sub['prediction'] = sub[['last_purchase', 'other_colors', 'popular_items']]\
    .apply(blend, w=[100, 10, 1], axis=1, k=32).apply(prune, ok_set=sold_set)
predicted = sub.prediction.apply(lambda s: [] if pd.isna(s) else s.split())
score = validation(actual, predicted, grouping, score, index='Prediction', ignore=(val_week == 105))

if val_week == 105: sub[['customer_id', 'prediction']].to_csv('./submissions/partitioned_validation.csv', index=False)