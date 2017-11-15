# coding: utf-8
# 717005
# 손실함수

import numpy as np

def MSE(y, t):
    loss = 0.5 * np.sum((y-t)**2)
    return loss

# 정답은 2
t = [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.2, 0.01, 0.5, 0.0, 0.02, 0.01, 0.0, 0.15, 0.1, 0.02]

loss1 = MSE(np.array(t), np.array(y1))
print(loss1) # 0.16175



y2 = [0.2, 0.01, 0.0, 0.5, 0.02, 0.01, 0.0, 0.15, 0.1, 0.02]

loss2 = MSE(np.array(t), np.array(y2))
print(loss2) # 0.16175