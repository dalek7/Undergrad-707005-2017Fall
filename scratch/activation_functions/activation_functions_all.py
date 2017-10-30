# coding: utf-8
# 717005
# 활성화 함수 (Activation functions) 직접 구현
# 수업 시간 구현 내용

import numpy as np
import matplotlib.pylab as plt

def my_step_function2(x):
    return np.array(x > 0, dtype=np.int)


def my_step_function(x):
    val = [] # Empty list
    for i in range(x.size):
        if x[i]>0 :
            val.append(1)
        else:
            val.append(0)
    return val

def my_relu_function(x):
    val = []  # Empty list
    for i in range(x.size):
        if x[i] > 0:
            val.append(x[i])
        else:
            val.append(0)
    return val

def my_relu_function2(x):
    return np.maximum(0, x)

def my_sigmoid(x):
    val = 1 / ( 1+ np.exp(-x))
    return val


X = np.arange(-5.0, 5.0, 0.1)
#tmp1 = np.zeros_like(X)
Y1 = my_step_function2(X)
Y2 = my_sigmoid(X)
Y3 = my_relu_function(X)

#print(X.size)
#print(tmp1)
#print(range(X.size))

plt.plot(X, Y1)
plt.plot(X, Y2)
plt.plot(X, Y3)
#plt.ylim(-0.1, 1.1)  # y축의 범위 지정

plt.show()