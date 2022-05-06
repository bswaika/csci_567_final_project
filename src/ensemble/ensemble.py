import os
import numpy as np
import pandas as pd
import gc

sub0 = pd.read_csv('./submissions/hm_0231.csv').sort_values('customer_id').reset_index(drop=True)
sub1 = pd.read_csv('./submissions/trending_prod_weekly.csv').sort_values('customer_id').reset_index(drop=True)
sub2 = pd.read_csv('./submissions/exponential_decay.csv').sort_values('customer_id').reset_index(drop=True)
# sub3 = pd.read_csv('./submissions/lstm_sequential.csv').sort_values('customer_id').reset_index(drop=True)
sub3 = pd.read_csv('./submissions/lstm_fix.csv').sort_values('customer_id').reset_index(drop=True)
sub4 = pd.read_csv('./submissions/hm_0224.csv').sort_values('customer_id').reset_index(drop=True)
sub5 = pd.read_csv('./submissions/time_friend.csv').sort_values('customer_id').reset_index(drop=True)
sub6 = pd.read_csv('./submissions/age_rule.csv').sort_values('customer_id').reset_index(drop=True)
# sub7 = pd.read_csv('./submissions/faster_trending.csv').sort_values('customer_id').reset_index(drop=True)
sub7 = pd.read_csv('./submissions/trending_prod.csv').sort_values('customer_id').reset_index(drop=True)

sub0.columns = ['customer_id', 'prediction0']
sub0['prediction1'] = sub1['prediction']
sub0['prediction2'] = sub2['prediction']
sub0['prediction3'] = sub3['prediction']
sub0['prediction4'] = sub4['prediction']
sub0['prediction5'] = sub5['prediction']
sub0['prediction6'] = sub6['prediction']
sub0['prediction7'] = sub7['prediction'].astype(str)

del sub1, sub2, sub3, sub4, sub5, sub6, sub7
gc.collect()
sub0.head()

def cust_blend(dt, W = [1,1,1,1,1,1,1,1]):
    #Global ensemble weights
    #W = [1.15,0.95,0.85]

    #Create a list of all model predictions
    REC = []

    # Second Try
    REC.append(dt['prediction0'].split())
    REC.append(dt['prediction1'].split())
    REC.append(dt['prediction2'].split())
    REC.append(dt['prediction3'].split())
    REC.append(dt['prediction4'].split())
    REC.append(dt['prediction5'].split())
    REC.append(dt['prediction6'].split())
    REC.append(dt['prediction7'].split())

    #Create a dictionary of items recommended.
    #Assign a weight according the order of appearance and multiply by global weights
    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += (W[M]/(n+1))
            else:
                res[v] = (W[M]/(n+1))

    # Sort dictionary by item weights
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())

    # Return the top 12 items only
    return ' '.join(res[:12])

sub0['prediction'] = sub0.apply(cust_blend, W = [0.8, 0.73, 0.8, 0.9, 0.69, 0.5, 0.9, 1.05], axis=1)

del sub0['prediction0']
del sub0['prediction1']
del sub0['prediction2']
del sub0['prediction3']
del sub0['prediction4']
del sub0['prediction5']
del sub0['prediction6']
del sub0['prediction7']
gc.collect()

sub1 = pd.read_csv('./submissions/partitioned_validation.csv').sort_values('customer_id').reset_index(drop=True)
sub1['prediction'] = sub1['prediction'].astype(str)

sub0.columns = ['customer_id', 'prediction0']
sub0['prediction1'] = sub1['prediction']

del sub1
gc.collect()

def cust_blend_2(dt, W = [1,1,1,1,1]):
    #Global ensemble weights
    #W = [1.15,0.95,0.85]

    #Create a list of all model predictions
    REC = []

    # Second Try
    REC.append(dt['prediction0'].split())
    REC.append(dt['prediction1'].split())

    #Create a dictionary of items recommended.
    #Assign a weight according the order of appearance and multiply by global weights
    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += (W[M]/(n+1))
            else:
                res[v] = (W[M]/(n+1))

    # Sort dictionary by item weights
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())

    # Return the top 12 items only
    return ' '.join(res[:12])
sub0['prediction'] = sub0.apply(cust_blend_2, W = [1.15, 0.75], axis=1)

del sub0['prediction0']
del sub0['prediction1']

sub0.to_csv('./submissions/submission_9.csv', index=False)