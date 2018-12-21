import tensorflow as tf

a=tf.Variable(tf.constant([[1.0,2,3],[4,5,6]]))
b=tf.Variable(tf.constant([[1.0,0,0],[0,1,0]]))

x = tf.nn.softmax_cross_entropy_with_logits(labels=b, logits=a)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print sess.run(a).shape
    print sess.run(b).shape
    print sess.run(x).shape
