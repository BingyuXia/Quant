import tensorflow as tf
import numpy as np
import matplotlib.pyplot
from tensorflow.examples.tutorials.mnist import input_data
#Hyper parameters
n_samples  =2000
batch_size = 100
epochs = 12
lr = 0.03
input_size = 
output_size = 
layers = [100]*5
Activation = tf.nn.relu
B_Init = 
W_Init =
batch_norm = True

#Training data

#tensorflow placeholder
X = tf.placeholder(tf.float32, [None, input_size])
y  = tf.placeholder(tf.float32, [None, output_size])
tf_is_train = tf.placeholder(tf.bool, None) #flag for using BN on training or testing

#Build layers
def add_layer(layer, outsize, pre_activation=pre_activation, ac=None, is_bn=False):
	layer = tf.layers.dense(layer, outsize, kernel_initializer=W_Init, bias_initializer=B_Init)
	pre_activation.append(layer)
	if is_bn:
		tf.layers.batch_normalization(layer, momentum=0.6, training=tf_is_train)
		output = layer if ac=None else ac(layer)
	return output
#Plot histogram
def plot_histogram(l_in, pre_ac):
	for i, (ax_pre, ax_in) in enumerate(zip(axs[0, :], axs[1, :])):
		[a.clear() for a in [ax_pre, ax_in]]
		ax_pre.set_title('L'+str(i)):
		ax_pre.hist(pre_ac[i].ravel(), bins=10,)
		ax_in.hist(pre_ac[i].ravel(), bins=10,)
	plt.pause(0.01)


pre_activation = [X]
if batch_norm:
	layer_input = [tf.layer.batch_normalization(X, training=tf_is_train)]
else:
	layer_input = [X]

for i in layers:
	layer_input.append(add_layer(layer=layer_input[-1], outsize=i, ac=Activation, is_bn=batch_norm))
output = tf.layers.dense(layer_input[-1], outsize, kernel_initializer=W_Init, bias_initializer=B_Init)
pred = tf.nn.softmax(output)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y))

# !! IMPORTANT !! the moving_mean and moving_variance need to be updated,
# pass the update_ops with control_dependencies to the train_op
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# f, axs = plt.subplot(2, len(layers)+1, figsize=(10, 5))
# plt.ion()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in epochs:
		#train
		sess.run([loss, optimizer], {X: train_x, y: train_y, tf_is_train:True})
		#test
		sess.run([layer_input, pre_activation], {X: test_x, y: test_y, tf_is_train:False})


# plt.ioff()


