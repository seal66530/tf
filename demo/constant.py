import tensorflow as tf

a=tf.constant(([[1.0,2.0],[5,6]]),name="a")
b=tf.constant(([3.0,4.0]),name="b")

c=a+b
print(c)

print(a.get_shape())
print(b.get_shape())
print(c.get_shape())

with tf.Session() as sess:
    print(sess.run(c))
