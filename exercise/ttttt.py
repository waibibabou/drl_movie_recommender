import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import time

np.random.seed(42)  # 设置numpy随机数种子
tf.random.set_seed(42)  # 设置tensorflow随机数种子

# 生成训练数据
x = np.linspace(-1, 1, 100)
x = x.astype('float32')
y = x * x + 1 + np.random.rand(100)*0.1  # y=x^2+1 + 随机噪声

x_train = np.reshape(x,[100,1])  # 将一维数据扩展为二维
y_train = np.reshape(y,[100,1])  # 将一维数据扩展为二维

# x_train = np.expand_dims(x, 1)  # 将一维数据扩展为二维
# y_train = np.expand_dims(y, 1)  # 将一维数据扩展为二维
plt.plot(x, y, '.')  # 画出训练数据


def create_model():
    inputs = keras.Input((1,))
    x = keras.layers.Dense(10, activation='relu')(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = create_model()  # 创建一个模型
loss_fn = keras.losses.MeanSquaredError()  # 定义损失函数
optimizer = keras.optimizers.SGD()  # 定义优化器


@tf.function  # 将训练过程转化为图执行模式
def train():
    with tf.GradientTape() as tape:
        y_pred = model(x_train, training=True)  # 前向传播，注意不要忘了training=True
        loss = loss_fn(y_train, y_pred)  # 计算损失
        tf.summary.scalar("loss", loss, epoch+1)  # 将损失写入tensorboard
    grads = tape.gradient(loss, model.trainable_variables)  # 计算梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 使用优化器进行反向传播
    return loss


epochs = 1000
begin_time = time.time()  # 训练开始时间
for epoch in range(epochs):
    loss = train()
    print('epoch:', epoch+1, '\t', 'loss:', loss.numpy())  # 打印训练信息
end_time = time.time()  # 训练结束时间

print("训练时长：", end_time-begin_time)

# 预测
y_pre = model.predict(x_train)

# 画出预测值
plt.plot(x, y_pre.squeeze())
plt.show()
