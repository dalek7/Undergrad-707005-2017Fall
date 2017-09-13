# -*- coding: utf-8 -*-
# 717005
import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1) # 0에서 6까지 0.1 간격으로 생성
print('x = ', x)
y = np.sin(x)
signal_length = y.shape[0]


noise = np.random.normal(0, 1, signal_length)
y2 = y + 0.2* noise

y3 = 2* x -0.3
y4 = y3 + noise


for i in range(signal_length):
    print("%f\t%f\t%f"% (x[i], y3[i], y4[i]))


# 그래프 그리기
plt.figure(1)
plt.plot(x, y)
plt.plot(x, y2)

# 새 figure 추가
plt.figure(2)
plt.plot(x, y3)
plt.plot(x, y4)

# 새 figure 추가
plt.figure(3)
plt.plot(x, y4,'ro-')
plt.show()


