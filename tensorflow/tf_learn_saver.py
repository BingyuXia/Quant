import tensorflow as tf
import numpy as np
##Save to file
W = tf.Variable(np.random.randint(6, size=(2,3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.random.randint(6, size=(3,)), dtype=tf.float32, name="biases")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

save = False
with tf.Session() as sess:
	sess.run(init)
	if save:
		save_path = saver.save(sess, "./model/model.ckpt")
		print(sess.run([W,b]))
		print("Save to path: %s" % save_path)
	else:
		saver.restore(sess, "./model/model.ckpt")
		print(sess.run([W,b]))  


