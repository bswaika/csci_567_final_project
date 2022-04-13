import pandas as pd
import numpy as np
from preprocess import ArticlePreProcModel
from tqdm import tqdm

NUM_ARTICLES = 1_001

def load_articles(filepath):
    articles_kw = pd.read_csv(filepath)
    keywords = articles_kw['keywords'].to_numpy()
    ids = articles_kw['article_id'].to_numpy()
    keywords = keywords.reshape(keywords.shape[0], 1)
    max_len = max(map(lambda x: len(x[0].split()), keywords))

    article_features = articles_kw[['keywords', 'article_id']].astype({'keywords': np.object0, 'article_id': np.int64})
    article_features = [np.array(v).reshape(NUM_ARTICLES, 1) for n, v in article_features.items()]
    
    return article_features, ids, keywords, max_len

def query_articles(article_ids, article_features, result):
    results = [[], []]
    for id in tqdm(article_ids, desc='processing articles'):
        condition = article_features[1][:, 0] == id
        if np.sum(condition) == 0:
            results[0].append(['null ' * 14])
            results[1].append([id])
        else:
            results[0].append(article_features[0][condition, 0].tolist())
            results[1].append(article_features[1][condition, 0].tolist())

    result.append(np.array(results[0], dtype=np.object0))
    result.append(np.array(results[1], dtype=np.int64))

def make_article_preprocessing_model(ids, keywords, max_len):
    return ArticlePreProcModel(max_len, keywords, ids, name='article_preprocessing') 

if __name__ == '__main__':
    data, i, k, m = load_articles('./data/articles_keywords.csv')
    ids = [882888002, 111565001, 123, 111565001, 706016002, 759871002]
    articles = query_articles(ids, data)
    print(articles)
    model = make_article_preprocessing_model(i, k, m)
    model.compile()
    print(model.predict(articles))