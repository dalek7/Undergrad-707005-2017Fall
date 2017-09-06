# 717005

import numpy as np

a = np.array([[1, 2], [3, 4]])

print(a)
print("===========")

print(np.mean(a)) # 2.5
print("===========")

print(np.mean(a, axis=0)) # [ 2.  3.]
print("===========")

print(np.mean(a, axis=1)) # [ 1.5  3.5]
print("===========")

# tuple index out of range
# print np.mean(a, axis=2)
