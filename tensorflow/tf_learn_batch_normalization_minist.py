import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data

#Load Data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#hyper paramaters
input_size=784
output_size=10
lr = 0.03
Activation = tf.nn.relu
B_Init = tf.constant_initializer(value = 0.0, dtype = tf.float32) 
W_Init = tf.random_normal_initializer(stddev=1.0, dtype = tf.float32)

layers = [600, 300, 100, 50]
batch_size = 100
epochs = 5

batch_norm = False


#tensorflow placeholder
X = tf.placeholder(tf.float32, [None, input_size])
y  = tf.placeholder(tf.float32, [None, output_size])
tf_is_train = tf.placeholder(tf.bool, None) #flag for using BN on training or testing


#Build layers
def add_layer(layer, outsize, ac=None, is_bn=False):
	layer = tf.layers.dense(layer, outsize, kernel_initializer=W_Init, bias_initializer=B_Init)
	pre_activation.append(layer)
	if is_bn:
		tf.layers.batch_normalization(layer, momentum=0.6, training=tf_is_train)
	output = layer if ac is None else ac(layer)
	return output


#Main
pre_activation = [X]
if batch_norm:
	layer_input = [tf.layers.batch_normalization(X, training=tf_is_train)]
else:
	layer_input = [X]

for i in layers:
	layer_input.append(add_layer(layer=layer_input[-1], outsize=i, ac=Activation, is_bn=batch_norm))
output = tf.layers.dense(layer_input[-1], output_size, kernel_initializer=W_Init, bias_initializer=B_Init)
pred = tf.nn.softmax(output)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
	total_batch = int(mnist.train.num_examples/batch_size)
	train_size = mnist.train.num_examples
	valid_size = mnist.test.num_examples
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		#train
		avg_cost = 0.
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			sess.run([loss, optimizer], {X: batch_x, y: batch_y, tf_is_train:True})
		#test
		l_in, pre_act = sess.run([layer_input, pre_activation], {X: mnist.test.images, tf_is_train:False})
	with open("layer_input.pkl","wb") as f:
		pickle.dump(l_in, f)
	with open("pre_activation.pkl","wb") as f:
		pickle.dump(pre_act, f)

