# 717005
import numpy as np

a = np.array([[1, -3, 2], [4, 0, -5]])
b = np.array([[3], [1], [2]])
c = np.matmul(a, b)

'''
a = np.array([[1, 2, 4], [2, 6, 0]])
b = np.array([[4, 1, 4, 3], [0, -1, 3, 1], [2, 7, 5, 2]])
c = np.matmul(a, b)
'''
print('a=')
print(a)
print('b=')
print(b)

print('ab=')
print(c)