import random

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras import Model
import numpy as np
from tensorflow import keras
import time

x=np.linspace(0,10,100)
y=x**2+np.random.rand(100)*0.1+1

x_train=np.reshape(x,[100,1])
y_train=np.reshape(y,[100,1])

# # 生成训练数据
# x = np.linspace(-1, 1, 100)
# x = x.astype('float32')
# y = x * x + 1 + np.random.rand(100)*0.1  # y=x^2+1 + 随机噪声
#
# x_train = np.reshape(x,[100,1])  # 将一维数据扩展为二维
# y_train = np.reshape(y,[100,1])  # 将一维数据扩展为二维

plt.plot(x, y, '.')  # 画出训练数据

def create_model():
    inputs=Input((1,))
    dense1=Dense(10,activation='sigmoid')(inputs)
    dense2=Dense(10,activation='sigmoid')(dense1)
    outputs=Dense(1)(dense2)

    model=Model(inputs=inputs,outputs=outputs)
    return model

my_model=create_model()
my_model.summary()
loss_fn=keras.losses.MeanSquaredError()
optimizer=keras.optimizers.SGD()

@tf.function
def train():
    with tf.GradientTape() as tape:
        y_pre=my_model(x_train,training=True)
        loss=loss_fn(y_train,y_pre)

    grads=tape.gradient(loss,my_model.trainable_variables)
    optimizer.apply_gradients(zip(grads,my_model.trainable_variables))
    return loss

epochs=5000
begin_time=time.time()
for epoch in range(epochs):
    loss=train()#模型的输入可以为ndarray类型，但是计算过程中以及最后结果都为tensor类型
    print(f'epoch:{epoch}\t loss:{loss}')
end_time=time.time()

y_pre=my_model.predict(x_train)
plt.plot(x,y_pre,label='result')
plt.legend()
plt.show()

