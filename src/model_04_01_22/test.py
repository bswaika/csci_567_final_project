import pandas as pd

df = pd.read_csv('../../dist/model_04_01_22/submission.csv')
df['prediction'] = df['prediction'].apply(lambda x: ' '.join(map(lambda y: f'0{y}', x.split(' '))))

df.to_csv('./submission.csv', index=False)