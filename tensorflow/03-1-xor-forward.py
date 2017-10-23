# coding: utf-8
# XOR
# 기존에 학습으로 구한 파라미터 값을 이용하여 forward만 계산함
# Seung-Chan Kim

import tensorflow as tf
import numpy as np


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

W1_data =[[ 6.26347065,  6.12451124],
          [-6.38764334, -5.81880665]]

W2_data = [[ 10.10004139],
          [ -9.59866238]]

b1_data = [-3.40124607,  2.879565  ]
b2_data = [ 4.46212006]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.placeholder(tf.float32, [2, 2])
b1 = tf.placeholder(tf.float32, [2])
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.placeholder(tf.float32, [2, 1])
b2 = tf.placeholder(tf.float32, [1])
y_forward = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(y_forward > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    y_calc, c, a= sess.run([y_forward, predicted, accuracy],
                        feed_dict={X: x_data, Y: y_data, W1:W1_data, b1:b1_data, W2:W2_data, b2:b2_data})

    print("Calc: ", y_calc)
    print("Correct: ", c)
    print("Accuracy: ", a*100.0)


