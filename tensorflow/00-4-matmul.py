# coding: utf-8
# 717005
# placeholder를 이용한 행렬 연산
import tensorflow as tf

x = tf.placeholder("float", None)
x2 = tf.placeholder(tf.float32, shape=[2, 2])

y1 = x * 2

c1 = tf.constant([[1.0, 1.0], [0.0, 1.0]])
y2 = tf.matmul(x2, c1)

with tf.Session() as session:
    result = session.run(y1, feed_dict={x: [1, 2, 3]})
    print(result)
    print('')
    result = session.run(y2, feed_dict={x2: [[3.0, 4.0], [5.0, 6.0]]})
    print(result)