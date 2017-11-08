# XOR
# based on https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-2-xor-nn.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1


x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

y_data = [[0],
          [1],
          [1],
          [0]]


x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

#x1 = x_data[[0], :]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost/loss function

cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        print(step)

    for i in range(4):
        x1 = x_data[[i], :]

        l1 = tf.sigmoid(tf.matmul(x1, W1) + b1)
        l2 = tf.sigmoid(tf.matmul(l1, W2) + b2)
        l2cast = tf.cast(l2 > 0.5, dtype=tf.float32)
        print i, sess.run(l2), sess.run(l2cast), y_data[[i], :]

