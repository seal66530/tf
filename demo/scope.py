import tensorflow as tf

with tf.variable_scope("foo"):
    v1 = tf.get_variable("v", [1], tf.float32, tf.constant_initializer(2.0))

with tf.variable_scope("foo", reuse=True):
    v2 = tf.get_variable("v", [1], tf.float32, tf.constant_initializer(2.0))

with tf.variable_scope("", reuse=True):
    v3 = tf.get_variable("foo/v", [1], tf.float32)

print(v1==v2)
print(v1==v3)
print(v1.name)
