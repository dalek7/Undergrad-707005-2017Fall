# 717005
# MNIST Visualization


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


img1 = mnist.test.images[200]
img1 = np.array(img1, dtype='float')
pixels = img1.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

