import tensorflow as tf

w1=tf.Variable(tf.random_normal((2,3), stddev=1, mean=5))
w2=tf.Variable(tf.random_normal((3,1), stddev=2, mean=2))
#w3=tf.Variable([1.0,2.0,3.0])

x=tf.constant([[0.7,0.9]])

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

init_op = tf.global_variables_initializer()
    
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    #print(sess.run(w3))
    print(sess.run(y))
