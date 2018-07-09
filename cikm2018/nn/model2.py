import tensorflow as tf

class Model:
    def __init__(self, word_emb_dim, rnn_hid_dim, rnn_n_layers, max_sen_length, learning_rate, keep_prob
                 , vocab_size, n_class):

        self.word_emb_dim = word_emb_dim
        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_n_layers = rnn_n_layers
        self.max_sen_length = max_sen_length
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.vocab_sze = vocab_size
        self.n_class = n_class

    def build(self):
        """
        building placeholder
        """
        self.input1, self.input2, self.sequence_lengths1, self.sequence_lengths2, self.label, self.lr, self.keep_prob = \
            self.__build_placeholder(init_learning_rate=self.learning_rate, init_keep_prob=self.keep_prob, max_sen_length=
            self.max_sen_length, n_class=self.n_class)

        """
        building word embedding
        """
        self.emb_1 = self.__build_embedding(scope='word_embedding',reuse=False,vocab_size=self.vocab_sze,word_emb_dim=self.word_emb_dim,
                                            input=self.input1)
        self.emb_2 = self.__build_embedding(scope='word_embedding',reuse=True,vocab_size=self.vocab_sze,word_emb_dim=self.word_emb_dim,
                                            input=self.input2)

        """
        building biGRU
        """
        (self.output_fw1, self.states_fw1), (self.output_bw1, self.states_bw1) = self.__build_GRU(scope='biGRU',
            rnn_n_layers=self.rnn_n_layers, rnn_hid_dim=self.rnn_hid_dim, keep_prob=self.keep_prob, reuse=False,
            sequence_lengths=self.sequence_lengths1, input=self.emb_1)

        (self.output_fw1, self.states_fw1), (self.output_bw1, self.states_bw1) = self.__build_GRU(scope='biGRU',
            rnn_n_layers=self.rnn_n_layers, rnn_hid_dim=self.rnn_hid_dim, keep_prob=self.keep_prob, reuse=False,
            sequence_lengths=self.sequence_lengths1, input=self.emb_1)

    def __build_GRU(self, scope, rnn_n_layers, rnn_hid_dim, keep_prob, reuse, sequence_lengths, input):
        with tf.variable_scope(scope, reuse=reuse):
            cells = [tf.contrib.rnn.BasicRNNCell(num_units=rnn_hid_dim) for layer in range(rnn_n_layers)]
            cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells]

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells_drop, cell_bw=cells_drop, inputs=input, sequence_length=sequence_lengths, dtype=tf.float32
            )

            output_fw, output_bw = outputs
            states_fw, states_bw = final_states

            return (output_fw, states_fw), (output_bw, states_bw)

    def __build_embedding(self, scope, vocab_size, word_emb_dim, reuse, input):
        with tf.variable_scope(scope, reuse=reuse):
            E = tf.get_variable(name='E',shape=[vocab_size, word_emb_dim],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            output = tf.nn.embedding_lookup(E,input)

        return output


    def __build_placeholder(self, init_learning_rate, init_keep_prob, max_sen_length, n_class):
        input1 = tf.placeholder(name='input1', dtype=tf.int32, shape=[None, max_sen_length])
        input2 = tf.placeholder(name='input2', dtype=tf.int32, shape=[None, max_sen_length])

        sequence_lengths1 = tf.placeholder(name='sequence_lengths_1', dtype=tf.int32, shape=[None])
        sequence_lengths2 = tf.placeholder(name='sequence_lengths_2', dtype=tf.int32, shape=[None])

        label  = tf.placeholder(name='label', dtype=tf.float32, shape=[None, n_class])

        lr = tf.placeholder_with_default(input=init_learning_rate, shape=[], name='learning_rate')
        keep_prob = tf.placeholder_with_default(input=init_keep_prob, shape=[], name='keep_prob')

        return input1, input2, sequence_lengths1, sequence_lengths2,  label, lr, keep_prob



if __name__ == '__main__':
    pass