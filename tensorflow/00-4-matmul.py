# 717005
import tensorflow as tf

x = tf.placeholder("float", None)
y = x * 2

x2 = tf.placeholder(tf.float32, shape=[2, 2])
y2 = tf.constant([[1.0, 1.0], [0.0, 1.0]])
z2 = tf.matmul(x2, y2)

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)
    result = session.run(y, feed_dict={x: [2, 5, -1]})
    print(result)
    print("----------------------------")
    z = session.run(z2, feed_dict={x2: [[3.0, 4.0], [5.0, 6.0]]})
    print(z2)
    print("----------------------------")
    print(z)
	
