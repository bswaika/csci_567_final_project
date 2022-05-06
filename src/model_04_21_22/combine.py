import os
import numpy as np
import pandas as pd
import gc

sub0 = pd.read_csv('./submission.csv').sort_values('customer_id').reset_index(drop=True)
sub1 = pd.read_csv('./submissions1.csv').sort_values('customer_id').reset_index(drop=True)

sub0.columns = ['customer_id', 'prediction0']
sub0['prediction1'] = sub1['prediction']

del sub1

def cust_blend(dt, W = [1,1]):
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

# sub0['prediction'] = sub0.apply(cust_blend, W = [1.15, 1.00], axis=1)
# sub0['prediction'] = sub0.apply(cust_blend, W = [1.00, 0.65], axis=1)
# sub0['prediction'] = sub0.apply(cust_blend, W = [1.00, 0.5], axis=1)
# sub0['prediction'] = sub0.apply(cust_blend, W = [1.00, 0.35], axis=1)
sub0['prediction'] = sub0.apply(cust_blend, W = [1.00, 0.6], axis=1)

del sub0['prediction0']
del sub0['prediction1']

sub0.to_csv('submission_merged_5.csv', index=False)