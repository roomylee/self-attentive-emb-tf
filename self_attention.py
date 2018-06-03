import tensorflow as tf


class SelfAttention:
    def __init__(self, sequence_length, num_classes,
                 vocab_size, embedding_size, hidden_size, d_a_size, r_size, fc_size, p_coef):
        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        text_length = self._length(self.input_text)

        # Embeddings
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Bidirectional(Left&Right) Recurrent Structure
        with tf.name_scope("bi-rnn"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.embedded_chars,
                                                                                       sequence_length=text_length,
                                                                                       dtype=tf.float32)
            self.H = tf.concat([self.output_fw, self.output_bw], axis=2)

        with tf.name_scope("self-attention"):
            self.W_s1 = tf.Variable(tf.random_uniform([2*hidden_size, d_a_size], -1.0, 1.0), name="W_s1")
            self.temp_mat = tf.nn.tanh(tf.einsum('aij,jk->aik', self.H, self.W_s1))
            self.W_s2 = tf.Variable(tf.random_uniform([d_a_size, r_size], -1.0, 1.0), name="W_s2")
            self.A = tf.nn.softmax(tf.einsum('aij,jk->aik', self.temp_mat, self.W_s2), name="attention")

        with tf.name_scope("sentence-embedding"):
            self.M = tf.einsum('aij,aik->ajk', self.H, self.A)

        with tf.name_scope("fully-connected"):
            self.M_flat = tf.reshape(self.M, shape=[-1, 2 * hidden_size * r_size])
            W_fc = tf.get_variable("W_fc", shape=[2 * hidden_size * r_size, fc_size],
                                   initializer=tf.contrib.layers.xavier_initializer())
            b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc")
            self.fc = tf.nn.xw_plus_b(self.M_flat, W_fc, b_fc, name="fc")

        with tf.name_scope("output"):
            W_output = tf.get_variable("W_output", shape=[fc_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_output")
            self.logits = tf.nn.xw_plus_b(self.fc, W_output, b_output, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        with tf.name_scope("penalization"):
            self.A_AT = tf.einsum('aij,aki->akj', self.A, tf.transpose(self.A, [0, 2, 1]))
            I = tf.ones([tf.shape(self.A)[0], 1]) * tf.expand_dims(tf.eye(r_size), 0)
            self.P = tf.square(tf.norm(self.A_AT - I, axis=[-2, -1], ord="fro"))
            self.loss_P = tf.reduce_mean(self.P * p_coef)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.loss_P

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")



    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length