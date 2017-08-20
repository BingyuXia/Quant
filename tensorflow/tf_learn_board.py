import tensorflow as tf
import numpy as np  
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    


def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_uniform([in_size, out_size]), name="weights")
	biases = tf.Variable(tf.zeros([out_size]) + 0.1, name="biases")

	layer = tf.add(tf.matmul(inputs, Weights), biases)
	if activation_function is None:
		outputs = layer
	else:
		outputs = activation_function(layer)

	return outputs

def add_layer_2(inputs, in_size, out_size, activation_function=None):
	weight_initializer = tf.random_normal_initializer(stddev=0.2, dtype = tf.float32)
	biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
	with tf.variable_scope("layer_01", ):
		Weights_01 = tf.get_variable("W", [in_size, 10], 
						initializer=weight_initializer)
		tf.summary.histogram("layer_01_weights", Weights_01)
		biases_01 = tf.get_variable("b", [10], initializer=biases_initializer)
		tf.summary.histogram("layer_01_biases", biases_01)

	with tf.variable_scope("layer_02", ):
		Weights_02 = tf.get_variable("W", [10, out_size], 
						initializer=weight_initializer)
		tf.summary.histogram("layer_02_weights", Weights_02)
		biases_02 = tf.get_variable("b", [out_size], initializer=biases_initializer)
		tf.summary.histogram("layer_02_weights", biases_02)

	with tf.name_scope("layer_01"):
		layer1 = tf.add(tf.matmul(inputs, Weights_01), biases_01)
		if activation_function is None:
			layer1 = layer1
		else:
			layer1 = activation_function(layer1)
	with tf.name_scope("layer_02"):
		outputs = tf.add(tf.matmul(layer1, Weights_02), biases_02)

	return outputs


x_data = np.linspace(-1, 1, 300).reshape(-1, 1)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
	X = tf.placeholder(tf.float32, [None, 1], name='x_input')
	Y = tf.placeholder(tf.float32, [None, 1], name='y_input')

prediction = add_layer_2(X, 1, 1, activation_function=tf.nn.relu)
#prediction = add_layer_2(l1, 10, 1, activation_function=None)
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(Y-prediction))
	tf.summary.scalar("loss", loss)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	
	writer = tf.summary.FileWriter("logs/", sess.graph)
	sess.run(init)
	for i in range(100):
		summary, l, _ = sess.run([merged, loss, train_step], 
			 		feed_dict={X: x_data[i*3:(i+1)*3], Y: y_data[i*3:(i+1)*3]})
		writer.add_summary(summary, i)
		if i % 10 == 0:
			print(l)
