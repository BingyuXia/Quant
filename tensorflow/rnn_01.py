#!/usr/bin/env python
#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


# 定义RNN
class RNN():
    def __init__(self):
        self.rnn_size = 100
        self.num_step = 10
        self.input_size = 60
        self.output_size = 5
        self.num_layers = 2
        self.batch_size = 100
        self.drop_prob = 0.5
        self.lr = 0.01
        self.X = tf.placeholder('float', [None, self.num_step, self.input_size])
        self.Y = tf.placeholder('float', [None, self.num_step, self.output_size])
        return

    def load_data(self):
        return

    def neural_network(self, model='lstm', drop=True):
        if model == 'rnn':
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fun = tf.nn.rnn_cell.BasicLSTMCell

        cell = cell_fun(self.rnn_size, state_is_tuple=True)
        if drop:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.drop_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

        initial_state = cell.zero_state(self.batch_size, tf.float32)
        softmax_w = tf.get_variable("softmax_w", [self.rnn_size, self.output_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.output_size], dtype=tf.float32)

        outputs, last_state = tf.nn.dynamic_rnn(cell, self.X, initial_state=initial_state,)
        output = tf.reshape(outputs, [-1, self.rnn_size])

        logits = tf.matmul(output, softmax_w) + softmax_b
        if drop:
            logits = tf.nn.dropout(logits, self.drop_prob)
        probs = tf.nn.softmax(logits)
        return logits, probs

    def train(self, opt=True, epochs=50):
        #Init:
        total_batches =

        if opt:
            #带权重的交叉熵：
            logits, probs = self.neural_network(drop=True)
            targets = tf.reshape(self.Y, [-1, self.output_size])
            loss = tf.nn.seq2seq.sequence_loss_by_example(
                     [logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],)
            cost = tf.reduce_mean(loss)

            #梯度缩放因子：
            #t_list[i] * clip_norm / max(global_norm, clip_norm)
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            max_grad_norm = 5  #?
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            train_op = optimizer.apply_gradients(zip(grads, tvars))

        else:
            cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost_func)


        epochs = 10
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(tf.all_variables())

            feed_dict = {}

            for epoch in range(epochs):
                #sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
                for batche in range(total_batches):
                    train_loss, _, _ = sess.run([cost, train_op],  \
                          feed_dict=feed_dict)

                if epoch % 7 == 0:
                    saver.save(sess, 'poetry.module', global_step=epoch)

        return