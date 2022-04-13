import tensorflow as tf

class RecSysModel(tf.keras.Model):
    def __init__(
        self, 
        dict_sizes, 
        embed_dims,
        dense_params, 
        **kwargs
        ):
        super().__init__(**kwargs)
        self._cust_embed = tf.keras.layers.Embedding(dict_sizes['customers'], embed_dims['customers'], input_length=1, name='customer_embedding')
        self._art_embed_id = tf.keras.layers.Embedding(dict_sizes['articles_ids'], embed_dims['articles'], input_length=1, name='article_embedding_id')
        self._art_embed_kw = tf.keras.layers.Embedding(dict_sizes['articles_kws'], embed_dims['articles'], input_length=17, name='article_embedding_keyword')

        self._cust_concat = tf.keras.layers.Concatenate(dtype=tf.float32, name='customer_concat')
        self._art_concat = tf.keras.layers.Concatenate(dtype=tf.float32, name='article_concat')
        self._concat = tf.keras.layers.Concatenate(dtype=tf.float32, name='similarity_concat')

        self._flatten = tf.keras.layers.Flatten()

        self._cust_dense = DenseModel(*dense_params['customers'], name='customer_dense')
        self._art_dense = DenseModel(*dense_params['articles'], name='article_dense')

        self._sim_network = DenseModel(*dense_params['similarity'], output_activation='sigmoid', name='similarity_network')
    
    def call(self, inputs):
        customers, articles = inputs[0], inputs[1]
        cid, cage = customers
        akw, aid = articles
        cide, akwe, aide = self._cust_embed(cid), self._art_embed_kw(akw), self._art_embed_id(aid)
        customer_net_input = self._cust_concat([self._flatten(cide), tf.cast(cage, tf.float32)])
        article_net_input = self._art_concat([self._flatten(akwe), self._flatten(aide)])
        c, a = self._cust_dense(customer_net_input), self._art_dense(article_net_input)
        similarity_net_input = self._concat([c, a])
        output = self._sim_network(similarity_net_input)
        return output


class DenseModel(tf.keras.Model):
    def __init__(self, num_layers, num_start_nodes, decay_factor, output_dim, output_activation='linear', **kwargs):
        super().__init__(**kwargs)
        self._layers = [tf.keras.layers.Dense(int(num_start_nodes * (decay_factor ** i)), activation=tf.nn.relu) for i in range(num_layers)]
        self._output = tf.keras.layers.Dense(output_dim, activation=output_activation)

    def call(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return self._output(x)

        