#-*-coding:utf-8_8_
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # GPU序号

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

X = tf.placeholder('float', [100])
Y = tf.placeholder('float', [100])

#create tensorflow structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))

weight_initializer = tf.radom
with tf.variabel_scope("test"):
	W = tf.get_variable('', [], initializer=weight_initializer)


#y = Weights*x_data +bias
# y = Weights * X + bias
y = tf.add(tf.matmul(Weights, X), bias)
loss = tf.reduce_mean(tf.square(y-Y))

lr = 0.5
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train, feed_dict={X:x_data, Y:y_data})
	if step % 20 == 0:
		print step, sess.run(Weights), sess.run(bias)