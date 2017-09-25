# coding: utf-8
# 717005
# placeholder를 이용한 행렬 연산
import tensorflow as tf

x  = tf.placeholder("float", None)
x2 = tf.placeholder(tf.float32, shape=[2, 2])

y = x * 2


with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)
    print('')
    result = session.run(y, feed_dict={x: [2, 5, -1]})
    print(result)
