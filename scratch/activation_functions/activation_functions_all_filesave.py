# coding: utf-8
# 717005

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def step(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)

y1 = step(x)
y2 = sigmoid(x)
y3 = relu(x)

np.savetxt('out/x.out', x, delimiter=',')
np.savetxt('out/y_step.out', y1, delimiter=',')
np.savetxt('out/y_sigmoid.out', y2, delimiter=',')
np.savetxt('out/y_relu.out', y3, delimiter=',')


plt.plot(x, y1, label='step')
plt.plot(x, y2, label='sigmoid')
plt.plot(x, y3, label='relu')

plt.xlim(-3.0, 2.5)
plt.ylim(-0.5, 2.5)

# Place a legend to the right of this smaller subplot.
plt.legend(loc=2, borderaxespad=0.)
plt.show()