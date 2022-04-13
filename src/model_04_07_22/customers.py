import pandas as pd
import numpy as np
from preprocess import CustomerPreProcModel
from tqdm import tqdm

NUM_CUSTOMERS = 300_001

def load_customers(filepath):
    customers = pd.read_csv(filepath)
    customers['age'].fillna(0, inplace=True)
    ids = customers['customer_id'].to_numpy()

    customer_features = customers[['customer_id', 'age']].astype({'age': np.float32, 'customer_id': np.object0})
    customer_features = [np.array(v).reshape(NUM_CUSTOMERS, 1) for n, v in customer_features.items()]

    return customer_features, ids

def query_customers(customer_ids, customer_features, result):
    results = [[], []]
    for id in tqdm(customer_ids, desc='processing customers'):
        condition = customer_features[0][:, 0] == id
        if np.sum(condition) == 0:
            results[0].append([id])
            results[1].append([0])
        else:
            results[0].append(customer_features[0][condition, 0].tolist())
            results[1].append(customer_features[1][condition, 0].tolist())

    result.append(np.array(results[0], dtype=np.object0))
    result.append(np.array(results[1], dtype=np.int64))

def make_customer_preprocessing_model(ids):
    return CustomerPreProcModel(ids, name='customer_preprocessing')

if __name__ == '__main__':
    data, i = load_customers('./data/customers.csv')
    ids = [
            '03d0011487606c37c1b1ed147fc72f285a50c05f00b9712e0fc3da400c864296',
            '6cc121e5cc202d2bf344ffe795002bdbf87178054bcda2e57161f0ef810a4b55',
            'e34f8aa5e7c8c258523ea3e5f5f13168b6c21a9e8bffccd515dd5cef56126efb',
            '3493c55a7fe252c84a9a03db338f5be7afbce1edbca12f3a908fac9b983692f2',
            '0bf4c6fd4e9d33f9bfb807bb78348cbf5c565846ff4006acf5c1b9aea77b0e54',
            'e6498c7514c61d3c24669f49753dc83fdff3ec1ba13902dd9184c959d8f0b249',
            'd80ed4ababfa96812e22b911629e6bcbf5093769051ea447e2b696ac98a3dae9',
            '1320d4b3dd6481cde05bb80fb7ca37397f70470b9afb96aeca5d41175acaf836',
            'a76cf5ea515d09f22b7fe3e8ea3c1944316bd6264a90e26cef126242ef3c5e11',
            '12312313kajsdnalb18237y1231l2i3b09yhpbu3190278g3h1p2aassn3388123',
            '03d0011487606c37c1b1ed147fc72f285a50c05f00b9712e0fc3da400c864296',
        ]
    customers = query_customers(ids, data)
    print(customers)
    model = make_customer_preprocessing_model(i)
    model.compile()
    print(model.predict(customers))