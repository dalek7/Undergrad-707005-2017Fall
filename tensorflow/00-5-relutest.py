# coding: utf-8
# 717005
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

x = tf.placeholder("float", None)
relu_out = tf.nn.relu(x)

# Launch the default graph.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 첫번째 테스트
    for i in range(10):
        relu_out1 = sess.run(relu_out, feed_dict={x:i-5})
        print (i-5, '-->', relu_out1)

    # 두번째 테스트
    xval = np.arange(-5.0, 5.0, 0.1)
    relu_out1 = sess.run(relu_out, feed_dict={x: np.array([xval])})
    print relu_out1

    plt.plot(xval, xval, '.')
    plt.plot(xval, relu_out1[0])
    plt.ylim(-5.0, 5.5)
    plt.show()

