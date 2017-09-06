# -*- coding: utf-8 -*-
# creating a tuple
months = ('January','February','March','April','May','June',\
'July','August','September','October','November','  December')

print(months)

# iterate through them:
# 하나씩 출력하기
for item in months:
    print item


t = ('john', 32, (2,3,4,5), 'hello')
print(t)
print(t[2])
print(t[2][1])
print(t[:2]) # index 포함 X
print(t[2:]) # index 포함 O

print(t[-1])
print(t[-2])
