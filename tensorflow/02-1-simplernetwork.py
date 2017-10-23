import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # reproducibility

x = tf.placeholder("float", [1, 3])
w = tf.Variable(tf.random_normal([3, 3]), name='w')
y = tf.matmul(x, w)


# Launch the default graph.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y1 = sess.run(y, feed_dict={x:np.array([[1.0, 2.0, 3.0]])})

    print('x=', [1.0, 2.0, 3.0])
    print('w=', sess.run(w))
    print('y1=', y1)