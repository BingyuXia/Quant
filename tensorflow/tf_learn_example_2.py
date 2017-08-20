import tensorflow as tf
import numpy as np

state = tf.Variable(0, name="counter")
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

matrix1 = tf.constant([[3., 3.]], tf.float32)
matrix2 = tf.constant([[2.],
					   [2.]], tf.float32)
const = tf.constant([1.])
product = tf.add(const, tf.matmul(matrix1, matrix2))

input1 = tf.placeholder('float', [2, 3])
input2 = tf.placeholder('float', [3, 2])

output = tf.matmul(input2 , input1 ) 


init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	out_data = sess.run(output, feed_dict={input1:np.random.rand(2, 3), input2:np.random.rand(3,2)})
	print(out_data)