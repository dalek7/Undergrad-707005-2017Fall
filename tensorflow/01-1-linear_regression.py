# -*- coding: utf-8 -*-
# 717005
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
import numpy as np
import matplotlib.pyplot as plt

# X and Y data
x_train = [1, 2, 3]
#y_train = [2, 4, 6] # 그냥 x_train 에 2배 곱해서 생성
y_train = [3, 5, 7]

#y_train = [2+0.1, 4-0.3, 6+0.15]

# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 2 and b should be 0
# But let TensorFlow figure it out

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

vcost =[] # 비어있는 array (list)

# Fit the line
x1 =np.linspace(np.min(x_train)-1, np.max(x_train)+1, num=5)


for step in range(2001):
    sess.run(train)
    vcost.append(sess.run(cost))

    if step % 20 == 0:
        w1 = sess.run(W)[0] # 기울기
        b1 = sess.run(b)[0] # bias
        print(step, sess.run(cost), w1, b1)


# 학습완료.
# Learns best fit W:[ 2.],  b:[ 0.]

w1 = sess.run(W)[0] # 기울기
b1 = sess.run(b)[0] # bias
str1 = 'y = ' + str(w1) +'x + ' + str(b1)
print(w1, b1)
print(str1)


plt.figure(1)
plt.plot(x_train, y_train,'o')
plt.plot(x1,w1*x1 + b1)
plt.grid() # 격자
#plt.axis((np.min(x_train) - 1, np.max(x_train) + 1, np.min(y_train) - 1, np.max(y_train) + 1))
plt.title(str1)


plt.figure(2)
plt.plot(vcost)
str2= "Cost = %.6f" % vcost[-1]
plt.title(str2)
plt.grid()
plt.show()
