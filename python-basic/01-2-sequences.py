# -*- coding: utf-8 -*-

li = ['hallym', 1, 3.141572, 'hello']
print(li)
li[1] = 45
print(li)
li.append('September')
print(li)

# 비어있는 list
v = []
for i in range(0,3):
    v.append(i)

print v

# 하나씩 출력하기
for item in v:
    print item