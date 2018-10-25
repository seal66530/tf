import tensorflow as tf

v1=tf.constant([1.,2,3,4])
v2=tf.constant([2.,1,5,0])

sess = tf.InteractiveSession()
print(tf.greater(v1,v2).eval())
print(sess.run(tf.where(tf.greater(v1,v2),v1,v2)))
sess.close()
