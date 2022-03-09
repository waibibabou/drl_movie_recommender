import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding,Dense,Input
from tensorflow import keras

def create_model():
    inputs=Input(shape=(2,),dtype=tf.int32)
    outputs=Embedding(100,6,input_length=2)(inputs)
    model=keras.Model(inputs=inputs,outputs=outputs)
    return model

model=create_model()

X_train=np.array([[2,2],[55,56],[56,55]])
y_pred=model(X_train)
print(y_pred)


x=np.array([[1,2,3],[2,2,3],[3,2,3]])
print(x.take(0,0))
print(x[0],x.shape,type(x))
x=tf.convert_to_tensor(x)
print(type(x))

x1=tf.constant([1,2,3],dtype='float32')
print(x1,type(x1))
x1=tf.reshape(x1,(1,3))
print(x1)

class test:
    a=0

    def __str__(self) -> str:
        return f"{self.a}"+'aaa'

test1=test()
print(test1)
