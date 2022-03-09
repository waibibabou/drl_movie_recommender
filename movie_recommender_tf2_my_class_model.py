import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf

import os
import pickle
import re
from tensorflow.python.ops import math_ops
import shutil
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import hashlib
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Embedding,Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,GlobalAveragePooling2D,Dense,Reshape
from tensorflow.keras import Model

def load_data():
    """
    Load Dataset from File
    """
    # 读取User数据
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python',
                        encoding='ISO-8859-1')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    # 改变User数据中性别和年龄
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    # 读取Movie数据集
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python',
                         encoding='ISO-8859-1')
    movies_orig = movies.values

    # 将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    # 电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)  # 向set中添加list的所有元素使用update函数

    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}

    # 将电影类型转成等长数字列表，长度是18
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])

    movies['Genres'] = movies['Genres'].map(genres_map)

    # 电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}

    # 将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

    movies['Title'] = movies['Title'].map(title_map)

    # 读取评分数据集
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python',
                          encoding='ISO-8859-1')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets_values = targets_pd.values

    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()


#嵌入矩阵的维度
embed_dim = 32
#用户ID个数
uid_max = max(features.take(0,1)) + 1 # 6040
#性别个数
gender_max = max(features.take(2,1)) + 1 # 1 + 1 = 2
#年龄类别个数
age_max = max(features.take(3,1)) + 1 # 6 + 1 = 7
#职业个数
job_max = max(features.take(4,1)) + 1# 20 + 1 = 21

#电影ID个数
movie_id_max = max(features.take(1,1)) + 1 # 3952
#电影类型个数
movie_categories_max = max(genres2int.values()) + 1 # 18 + 1 = 19
#电影名单词个数
movie_title_max = len(title_set) # 5216

#对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

#电影名长度
sentences_size = title_count # = 15
#文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
#文本卷积核数量
filter_num = 8

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}

# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20


np.set_printoptions(threshold=np.inf)

class MyNetwork(Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        #用户特征处理
        self.uid_embed_layer=Embedding(input_dim=uid_max,output_dim=32,input_length=1)
        self.gender_embed_layer=Embedding(input_dim=gender_max,output_dim=16,input_length=1)
        self.age_embed_layer=Embedding(input_dim=age_max,output_dim=16,input_length=1)
        self.job_embed_layer=Embedding(input_dim=job_max,output_dim=16,input_length=1)

        # 提取用户特征 第一层全连接
        self.uid_fc_layer=Dense(32,activation="relu")
        self.gender_fc_layer=Dense(32,activation="relu")
        self.age_fc_layer=Dense(32,activation="relu")
        self.job_fc_layer=Dense(32,activation="relu")

        # 第二层全连接
        # self.user_combine_layer1=tf.keras.layers.concatenate([self.uid_fc_layer,self.gender_fc_layer,self.age_fc_layer,self.job_fc_layer],2)
        self.user_combine_layer2=Dense(200,activation="relu")
        self.user_combine_layer_flat=Reshape([200])

        #电影特征处理
        self.movie_id_embed_layer=Embedding(movie_id_max,32,input_length=1)

        self.movie_categories_embed_layer1=Embedding(movie_categories_max,32,input_length=18)
        self.movie_categories_embed_layer2=tf.keras.layers.Lambda(lambda layer:tf.reduce_sum(layer,axis=1,keepdims=True))

        # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
        self.movie_title_embed_layer=Embedding(movie_title_max,32,input_length=15)
        # sp=self.movie_title_embed_layer.output_shape
        self.movie_title_embed_layer_expand=Reshape([15, 32, 1])

        # 对文本嵌入层使用4种不同尺寸的卷积核做卷积和最大池化
        self.conv_layer1=Conv2D(filter_num,(2,32),1,activation="relu")
        self.maxpool_layer1=MaxPool2D(pool_size=(sentences_size-2+1,1),strides=1)

        self.conv_layer2 = Conv2D(filter_num, (3, 32), 1, activation="relu")
        self.maxpool_layer2 = MaxPool2D(pool_size=(sentences_size - 3 + 1, 1), strides=1)

        self.conv_layer3 = Conv2D(filter_num, (4, 32), 1, activation="relu")
        self.maxpool_layer3 = MaxPool2D(pool_size=(sentences_size - 4 + 1, 1), strides=1)

        self.conv_layer4 = Conv2D(filter_num, (5, 32), 1, activation="relu")
        self.maxpool_layer4 = MaxPool2D(pool_size=(sentences_size - 5 + 1, 1), strides=1)

        # self.pool_layer=tf.keras.layers.concatenate([self.maxpool_layer1,self.maxpool_layer2,self.maxpool_layer3,self.maxpool_layer4],3)
        max_num=len(window_sizes)*filter_num

        self.pool_layer_flat = tf.keras.layers.Reshape([1, max_num])

        self.dropout_layer = tf.keras.layers.Dropout(dropout_keep)

        #电影特征第一层全连接
        self.movie_id_fc_layer=Dense(32,activation="relu")
        self.movie_categories_fc_layer=Dense(32,activation="relu")

        #第二层全连接
        # self.movie_combine_layer1=tf.keras.layers.concatenate([self.movie_id_fc_layer,self.movie_categories_fc_layer,self.dropout_layer],2)
        self.movie_combine_layer2=Dense(200,activation="relu")

        self.movie_combine_layer_flat=Reshape([200])

        #将用户特征与电影特征作为输入，经过两层全连接得到预测评分
        # self.inference_layer=tf.keras.layers.concatenate([self.user_combine_layer_flat,self.movie_combine_layer_flat],1)
        self.inference_dense=Dense(64,kernel_regularizer=tf.nn.l2_loss,activation="relu")
        self.inference=Dense(1)



    def call(self,x):
        print(f"调用call函数时输入x为{x}")
        uid=x[0]
        gender=x[1]
        age=x[2]
        job=x[3]
        movieid=x[4]
        moviece=x[5]
        movietitle=[6]

        uid=self.uid_embed_layer(uid)
        gender=self.gender_embed_layer(gender)
        age=self.age_embed_layer(age)
        job=self.job_embed_layer(job)

        uid=self.uid_fc_layer(uid)
        gender=self.gender_fc_layer(gender)
        age=self.age_fc_layer(age)
        job=self.job_fc_layer(job)

        #合并操作写在哪?
        user_feature=tf.concat([uid,gender,age,job],2)
        user_feature=self.user_combine_layer2(user_feature)
        user_feature=self.user_combine_layer_flat(user_feature)


        movieid=self.movie_id_embed_layer(movieid)
        moviece=self.movie_categories_embed_layer1(moviece)
        moviece=self.movie_categories_embed_layer2(moviece)

        movietitle=self.movie_title_embed_layer(movietitle)
        movietitle=self.movie_title_embed_layer_expand(movietitle)

        movietitle1=self.maxpool_layer1(self.conv_layer1(movietitle))
        movietitle2=self.maxpool_layer2(self.conv_layer2(movietitle))
        movietitle3=self.maxpool_layer3(self.conv_layer3(movietitle))
        movietitle4=self.maxpool_layer4(self.conv_layer4(movietitle))

        movietitle_feature=tf.concat([movietitle1,movietitle2,movietitle3,movietitle4],3)
        movietitle_feature=self.pool_layer_flat(movietitle_feature)
        movietitle_feature=self.dropout_layer(movietitle_feature)

        movieid=self.movie_id_fc_layer(movieid)
        moviece=self.movie_categories_fc_layer(moviece)

        movie_feature=tf.concat([movieid,moviece,movietitle_feature],2)
        movie_feature=self.movie_combine_layer2(movie_feature)
        movie_feature=self.movie_combine_layer_flat(movie_feature)

        combine_feature=tf.concat([user_feature,movie_feature],1)
        combine_feature=self.inference_dense(combine_feature)
        inference=self.inference(combine_feature)
        return inference


model=MyNetwork()

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

train_X,test_X,train_y,test_y=train_test_split(features,targets_values,test_size=0.2,random_state=0)

train_X=tf.convert_to_tensor(train_X,tf.float32)
test_X=tf.convert_to_tensor(test_X,tf.float32)
train_y=tf.convert_to_tensor(train_y,tf.float32)
test_y=tf.convert_to_tensor(test_y,tf.float32)


history=model.fit(train_X,train_y,batch_size=batch_size,epochs=5,validation_data=(test_X,test_y),validation_freq=1)

model.summary()


loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(loss,label='training loss')
plt.legend()
plt.show()

plt.plot(val_loss,label='test loss')
plt.legend()
plt.show()






