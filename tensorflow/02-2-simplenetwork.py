import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # reproducibility

x = tf.placeholder("float", [1, 3])
w = tf.Variable(tf.random_normal([3, 3]), name='w')
y = tf.matmul(x, w)
relu_out = tf.nn.relu(y)


# Launch the default graph.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('x=', [1.0, 2.0, 3.0])
    print('w=', sess.run(w))
    relu_out1 = sess.run(relu_out, feed_dict={x:np.array([[1.0, 2.0, 3.0]])})
    print('relu_out=', relu_out1)