# MNIST and NN
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as img


from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("/tmp/tf/MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# 테스트 하기
# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
# test set에서 랜덤으로 하나 골라 테스트
r = random.randint(0, mnist.test.num_examples - 1)
print("Random pick : ", r)
print(" (Label) = ", mnist.test.labels[r:r + 1])

print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys')#, interpolation='nearest'


# 내가 그린 숫자를 읽어오는 부분
# Custom images



x = img.imread("../data/4_28x.png")
x = x.astype(float)
x1 = np.array(x)
x1 = x1/255.0
x1 = 1.0 - x1
x1_flat = x1.reshape(1,784 ) # 28x28 이미지를 1x784로 변형하기

print("Label: ", 4)
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X:x1_flat}))
fig3 = plt.figure()

plt.imshow(x1_flat.reshape(28, 28),  cmap='Greys')



plt.show()