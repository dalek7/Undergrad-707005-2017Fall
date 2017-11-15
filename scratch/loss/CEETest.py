# coding: utf-8
# 717005
# 손실함수

import numpy as np

# Cross entropy error

def CEE(y, t):
    epsilon = 0.000000001
    v_temp = t * np.log(y + epsilon)
    loss = -np.sum(v_temp)
    return loss

# 정답은 두번째것
t  = [ 0, 1]
y1 = [ 0.2, 0.8]
y2 = [ 0.8, 0.2]
y3 = [ 0.1, 0.9]
y4 = [ 1.0, 0.0]

loss1 = CEE(np.array(t), np.array(y1))
print(loss1)

loss2 = CEE(np.array(t), np.array(y2))
print(loss2)

loss3 = CEE(np.array(t), np.array(y3))
print(loss3)

loss4 = CEE(np.array(t), np.array(y4))
print(loss4)