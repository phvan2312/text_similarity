import tensorflow as tf

class Model:
    def __init__(self, word_emb_dim, rnn_hid_dim, rnn_n_layers, max_sen_length, learning_rate, keep_prob
                 , vocab_size, n_class):

        self.word_emb_dim = word_emb_dim
        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_n_layers = rnn_n_layers
        self.max_sen_length = max_sen_length
        self.init_learning_rate = learning_rate
        self.init_keep_prob = keep_prob
        self.vocab_sze = vocab_size
        self.n_class = n_class

    def build(self, build_session = True):
        tf.reset_default_graph()

        """
        building placeholder
        """
        self.input1, self.input2, self.sequence_lengths1, self.sequence_lengths2, self.label, self.lr, self.keep_prob = \
            self.__build_placeholder(init_learning_rate=self.init_learning_rate, init_keep_prob=self.init_keep_prob, max_sen_length=
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

        (self.output_fw2, self.states_fw2), (self.output_bw2, self.states_bw2) = self.__build_GRU(scope='biGRU',
                                                                                                  rnn_n_layers=self.rnn_n_layers, rnn_hid_dim=self.rnn_hid_dim, keep_prob=self.keep_prob, reuse=True,
                                                                                                  sequence_lengths=self.sequence_lengths2, input=self.emb_2)

        self.state1 = tf.concat([self.states_fw1[1], self.states_bw1[1]], axis=1)
        self.state2 = tf.concat([self.states_fw2[1], self.states_bw2[1]], axis=1)

        """
        buildindg matching
        """
        self.scores = self.__build_matching(scope="matching", n_class=self.n_class, reuse=False, input1=self.state1, input2=self.state2)


        """
        building loss function and training op
        """
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,labels=self.label,name='softmax_loss')

        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr)

        if build_session:
            self.sess = tf.Session()
            init_op = tf.global_variables_initializer()

            self.sess.run(init_op)

    def __build_matching(self, scope, n_class, reuse, input1, input2):
        last_shape = input1.get_shape().as_list()[-1]

        with tf.variable_scope(scope, reuse=reuse):
            subtract = input2 - input1
            multiply = input2 * input1

            W_1 = tf.get_variable(name='W1', dtype=tf.float32, shape=[last_shape, last_shape],
                                  initializer=tf.contrib.layers.xavier_initializer())
            W_2 = tf.get_variable(name='W2', dtype=tf.float32, shape=[last_shape, last_shape],
                                  initializer=tf.contrib.layers.xavier_initializer())

            W_3 = tf.get_variable(name='W3', dtype=tf.float32, shape=[last_shape, last_shape],
                                  initializer=tf.contrib.layers.xavier_initializer())
            W_4 = tf.get_variable(name='W4', dtype=tf.float32, shape=[last_shape, last_shape],
                            initializer=tf.contrib.layers.xavier_initializer())

            V = tf.get_variable(name='V', dtype=tf.float32, shape=[last_shape, n_class],
                                  initializer=tf.contrib.layers.xavier_initializer())

            scores = tf.matmul(tf.nn.tanh(W_1 * input1 + W_2 * input2 + W_3 * subtract + W_4 * multiply), V, name='scores')

            return scores

    def __build_GRU(self, scope, rnn_n_layers, rnn_hid_dim, keep_prob, reuse, sequence_lengths, input):
        with tf.variable_scope(scope, reuse=reuse):
            cells = [tf.contrib.rnn.BasicRNNCell(num_units=rnn_hid_dim) for layer in range(rnn_n_layers)]
            cells_drop = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells])

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
    """
    word_emb_dim, rnn_hid_dim, rnn_n_layers, max_sen_length, learning_rate, keep_prob, vocab_size, n_class
    """
    params = {
        'word_emb_dim' : 300,
        'rnn_hid_dim': 64,
        'rnn_n_layers': 2,
        'max_sen_length': 100,
        'learning_rate': 0.001,
        'keep_prob': 0.9,
        'vocab_size': 10000,
        'n_class':2
    }

    model = Model(**params)
    model.build()