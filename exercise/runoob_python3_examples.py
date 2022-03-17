import math
import random
import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3], dtype=tf.float32)
print(a, type(a), a[0])

b = np.array([1, 2, 3], dtype='float32')
print(type(b))
b = b.astype('float64')
b = tf.convert_to_tensor(b)
print(b, b[0])
b = b.numpy()
print(b)
print('\n\n\n')

c = tf.constant([1, 2, 3], dtype=tf.float32)
print(c, c.numpy())
d=tf.cast(c, dtype=tf.int64)
print(c,d)

