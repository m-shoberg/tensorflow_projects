# importing tensorflow
import tensorflow as tf

tf.version.VERSION

# creating nodes in computation graph
node1 = tf.constant(3, dtype = tf.int32)
node2 = tf.constant(5, dtype = tf.int32)
node3 = tf.add(node1, node2)

# create tensorflow session object
#sess = tf.Session()

# evaluating node3 and printing the result
#print("Sum of node1 and node2 is:", sess.run(node3))
tf.print("Sum of node1 and node2 is:", node3)

# closing the session
#sess.close()

'''
msg = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(msg))
'''

'''
msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)
'''