# coding: utf-8
# 717005
# based on https://github.com/WegraLee/deep-learning-from-scratch/blob/master/ch03/relu.py
import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, x, '.')
plt.plot(x, y)
plt.ylim(-5.0, 5.5)
plt.show()