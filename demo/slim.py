import tensorflow as tf
import tensorflow.contrib.slim as slim

input = tf.Variable(tf.random_normal([100,100,100,3]))
net = slim.conv2d(input, 16, [3,3], scope='conv1')

writer=tf.summary.FileWriter("./slim_log",tf.get_default_graph())
writer.close()

