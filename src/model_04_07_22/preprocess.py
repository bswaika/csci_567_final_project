import tensorflow as tf

class ArticlePreProcModel(tf.keras.Model):
    def __init__(self, max_len, keywords, ids, **kwargs):
        super().__init__(**kwargs)
        self._vectorize = tf.keras.layers.TextVectorization(standardize=None, output_sequence_length=max_len)
        self._vectorize.adapt(keywords[:, 0])
        self._lookup = tf.keras.layers.IntegerLookup()
        self._lookup.adapt(ids)

        self.keyword_dict = self._vectorize.get_vocabulary()
        self.article_dict = self._lookup.get_vocabulary()

    def call(self, inputs):
        outputs = [self._vectorize(inputs[0]), self._lookup(inputs[1])]
        return outputs

class CustomerPreProcModel(tf.keras.Model):
    def __init__(self, ids, **kwargs):
        super().__init__(**kwargs)
        self._lookup = tf.keras.layers.StringLookup()
        self._lookup.adapt(ids)

        self.customer_dict = self._lookup.get_vocabulary()
    
    def call(self, inputs):
        outputs = [self._lookup(inputs[0]), inputs[1]]
        return outputs
