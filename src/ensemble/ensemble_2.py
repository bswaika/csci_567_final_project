import os
import numpy as np
import pandas as pd
import gc

sub0 = pd.read_csv('./submissions/submission_1.csv').sort_values('customer_id').reset_index(drop=True)
sub1 = pd.read_csv('./submissions/lstm_fix.csv').sort_values('customer_id').reset_index(drop=True)
sub2 = pd.read_csv('./submissions/trending_prod.csv').sort_values('customer_id').reset_index(drop=True)
sub3 = pd.read_csv('./submissions/byfonechris_combo.csv').sort_values('customer_id').reset_index(drop=True)
sub4 = pd.read_csv('./submissions/funksvd_recalls.csv').sort_values('customer_id').reset_index(drop=True)

sub0.columns = ['customer_id', 'prediction0']
sub0['prediction1'] = sub1['prediction']
sub0['prediction2'] = sub2['prediction']
sub0['prediction3'] = sub3['prediction']
sub0['prediction4'] = sub4['prediction']

del sub1, sub2, sub3, sub4
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

sub0['prediction'] = sub0.apply(cust_blend, W = [1.15, 0.4, 0.6, 0.5, 0.3], axis=1)

del sub0['prediction0']
del sub0['prediction1']
del sub0['prediction2']
del sub0['prediction3']
del sub0['prediction4']
gc.collect()

sub0.to_csv('./submissions/submission_6.csv', index=False)
