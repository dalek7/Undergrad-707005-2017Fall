# coding: utf-8
import numpy as np


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 다음의 조합을 넣어 동작 확인
xs = [(0, 0), (1, 0), (0, 1), (1, 1)]

x = (0,0)
y = OR(x[0], x[1])
print(str(x) , '-->', y)