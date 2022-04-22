import tensorflow as tf

hello = tf.constant('Hello, tensorflow')

# 打印当前tf的版本号
print("###########version###########", tf.__version__)

# 返回True，说明是可用的
print("++++++++++is_gpu_available+++++++++++", tf.test.is_gpu_available())
