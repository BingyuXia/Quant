import tensorflow as tf
import numpy as np

input_size = 
lr =

X = tf.placeholder(tf.float32, [None, input_size])

#Encoder
en_layers = []
en0 = tf.layers.dense(X, en_layers[0], tf.nn.tanh)
en1 = tf.layers.dense(en0, en_layers[1], tf.nn.tanh)
en2 = tf.layers.dense(en1, en_layers[2], tf.nn.tanh)
encoded = tf.layers.dense(en2, en_layers[3])

#Decoder
de_layers = []
de0 = tf.layers.dense(encoded, de_layers[0], tf.nn.tanh)
de1 = tf.layers.dense(de0, de_layres[1], tf.nn.tanh)
de2 = tf.layers.dense(de1, de_layers[2], tf.nn.tanh)
decoded = tf.layers.dense(de2, de_layers3[3], tf.nn.tanh)

loss = tf.losses.mean_squared_error(labels=X, predictions=decoded)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
