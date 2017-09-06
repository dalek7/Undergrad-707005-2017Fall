# 717005
# based on http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays
import numpy as np

a = np.array([[1., 2., 3.],[4.,5.,6.]])
print a
print a.shape
print

b = np.empty_like(a)
print b
print b.shape
print

c = np.zeros_like(a)
print c
print c.shape
print

d = np.ones_like(a)
print d
print d.shape

