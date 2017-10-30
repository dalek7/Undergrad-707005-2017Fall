# coding: utf-8
# 717005 수업시간에 구현한 내용
# AND Gate
# based on https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-2-xor-nn.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

# 다음의 정보를 컴퓨터에 알려줌
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

y_data = [[0],
          [0],
          [0],
          [1]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

# x1 = x_data[[0], :]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 1]), name='weight1')
b1 = tf.Variable(tf.random_normal([1]), name='bias1')
hypothesis = tf.sigmoid(tf.matmul(X, W1) + b1)

# cost/loss function
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis)) # 나중에...
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

vcost=[]
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        vcost.append(sess.run(cost, feed_dict={X: x_data, Y: y_data}))


    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

    W1val = sess.run(W1)
    b1val = sess.run(b1)

    print ('W1 = ', sess.run(W1))
    print ('b1 = ', sess.run(b1))


    for i in range(4):
        x1 = x_data[[i], :]
        # print x1
        l1 = tf.sigmoid(tf.matmul(x1, W1) + b1)
        l2cast = tf.cast(l1 > 0.5, dtype=tf.float32)
        print (i, sess.run(l1), sess.run(l2cast), y_data[[i], :])

plt.plot(vcost)
plt.show()