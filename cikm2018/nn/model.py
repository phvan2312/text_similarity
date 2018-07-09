import tensorflow as tf

class Model:
    def __init__(self, spa_max_sen_length, eng_max_sen_length, learning_rate, keep_prob, word_emb_dim, rnn_hid_dim,
                 vocab_size, rnn_n_layers):
        self.spa_max_sen_length = spa_max_sen_length
        self.eng_max_sen_length = eng_max_sen_length
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.word_emb_dim = word_emb_dim
        self.rnn_hid_dim = rnn_hid_dim
        self.vocab_size = vocab_size
        self.rnn_n_layers = rnn_n_layers

    def __build_placeholder(self, spa_max_sen_length, eng_max_sen_length, init_learing_rate, init_keep_prob):
        spa_input_1 = tf.placeholder(name='spa_input_1', dtype=tf.int32, shape=[None, spa_max_sen_length])
        spa_input_2 = tf.placeholder(name='spa_input_2', dtype=tf.int32, shape=[None, spa_max_sen_length])
        spa_sen_length_1 = tf.placeholder(name='spa_sen_length_1', dtype=tf.int32, shape=[None])
        spa_sen_length_2 = tf.placeholder(name='spa_sen_length_2', dtype=tf.int32, shape=[None])

        eng_input_1 = tf.placeholder(name='eng_input_1', dtype=tf.int32, shape=[None, eng_max_sen_length])
        eng_input_2 = tf.placeholder(name='eng_input_2', dtype=tf.int32, shape=[None, eng_max_sen_length])
        eng_sen_length_1 = tf.placeholder(name='eng_sen_length_1', dtype=tf.int32, shape=[None])
        eng_sen_length_2 = tf.placeholder(name='eng_sen_length_2', dtype=tf.int32, shape=[None])

        learning_rate = tf.placeholder_with_default(input=init_learing_rate,shape=[],name='learning_rate')
        keep_prob = tf.placeholder_with_default(input=init_keep_prob,shape=[],name='keep+prob') # init_keep_prob should be set to 1
        label = tf.placeholder(name='label', shape=[None, 2], dtype=tf.float32)

        return spa_input_1, spa_input_2, spa_sen_length_1, spa_sen_length_2, \
               eng_input_1, eng_input_2, eng_sen_length_1, eng_sen_length_2, \
               learning_rate, keep_prob, label



    def build(self):
        """
        build placeholders
        """
        self.spa_input_1, self.spa_input_2, self.spa_sen_length_1, self.spa_sen_length_2, self.eng_input_1, self.eng_input_2, \
        self.eng_sen_length_1, self.eng_sen_length_2, self.learning_rate, self.keep_prob, self.label = self.__build_placeholder(
            spa_max_sen_length=self.spa_max_sen_length, eng_max_sen_length=self.eng_max_sen_length,
            init_learing_rate=self.learning_rate, init_keep_prob=self.keep_prob
        )
        self.batch_size = tf.shape(self.spa_input_1)[0]

        """
        build word embedding
        """
        with tf.variable_scope('word_embedding'):
            with tf.variable_scope('spa'):
                self.spa_E = tf.get_variable(name='E',shape=[self.vocab_size, self.word_emb_dim],dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
                self.spa_emb_1 = tf.nn.embedding_lookup(self.spa_E, self.spa_input_1)
                self.spa_emb_2 = tf.nn.embedding_lookup(self.spa_E, self.spa_input_2)

            with tf.variable_scope('eng'):
                self.eng_E = tf.get_variable(name='E', shape=[self.vocab_size, self.word_emb_dim], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                self.eng_emb_1 = tf.nn.embedding_lookup(self.eng_E, self.eng_input_1)
                self.eng_emb_2 = tf.nn.embedding_lookup(self.eng_E, self.eng_input_2)

        """
        build encoder-decoder in NMT
        """
        with tf.variable_scope('encoder_decoder'):
            with tf.variable_scope('encoder_spa'):
                cells = [tf.contrib.rnn.BasicRNNCell(num_units=self.rnn_hid_dim) for layer in range(self.rnn_n_layers)]
                cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob) for cell in cells]

                self.spa_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
                zero_state = self.spa_multi_layer_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                self.spa_rnn_outputs_1, self.spa_states_1 = tf.nn.dynamic_rnn(cell=self.spa_multi_layer_cell,
                                                                              inputs=self.spa_input_1,
                                                                              sequence_length=self.spa_sen_length_1,
                                                                              initial_state=zero_state,
                                                                              swap_memory=True)

                self.spa_rnn_outputs_2, self.spa_states_2 = tf.nn.dynamic_rnn(cell=self.spa_multi_layer_cell,
                                                                              inputs=self.spa_input_2,
                                                                              sequence_length=self.spa_sen_length_2,
                                                                              initial_state=zero_state,
                                                                              swap_memory=True)

            with tf.variable_scope('decoder_eng'):
                cells = [tf.contrib.rnn.BasicRNNCell(num_units=self.rnn_hid_dim) for layer in range(self.rnn_n_layers)]
                cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob) for cell in cells]

                self.eng_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)

                self.eng_rnn_outputs_1, self.eng_states_1 = tf.nn.dynamic_rnn(cell=self.eng_multi_layer_cell,
                                                                              inputs=self.eng_input_1,
                                                                              sequence_length=self.eng_sen_length_1,
                                                                              initial_state=self.spa_states_1,
                                                                              swap_memory=True)

                self.eng_rnn_outputs_2, self.eng_states_2 = tf.nn.dynamic_rnn(cell=self.eng_multi_layer_cell,
                                                                              inputs=self.eng_input_2,
                                                                              sequence_length=self.eng_sen_length_2,
                                                                              initial_state=self.spa_states_2,
                                                                              swap_memory=True)






if __name__ == '__main__':
    print("Hello world");