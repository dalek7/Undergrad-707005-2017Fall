# coding: utf-8
# 717005
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Numpy version
mu1, sigma1 = 0, 1.0 # mean and standard deviation
gaussian_numbers1 = np.random.normal(mu1, sigma1, 1000)


# TensorFlow version
gaussian_numbers2 = tf.random_normal(shape=[1000], mean=mu1, stddev=sigma1, dtype=tf.float32)
# Launch the default graph.
sess = tf.Session()
gaussian_numbers2tf = sess.run(gaussian_numbers2)
sess.close()

# Figures - Numpy
plt.figure(1)
num_bins = 50
plt.hist(gaussian_numbers1, num_bins)
plt.title("Gaussian Histogram using np.random.normal()")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.figure(2)
plt.plot(gaussian_numbers1)
plt.title("np.random.normal()")

# Figures - Tensorflow
plt.figure(3)
num_bins = 50
plt.hist(gaussian_numbers2tf, num_bins)
plt.title("Gaussian Histogram using tf.random_normal()")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.figure(4)
plt.plot(gaussian_numbers2tf)
plt.title("tf.random_normal()")
plt.show()


