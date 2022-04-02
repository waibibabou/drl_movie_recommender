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
import datetime
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
import time
from tensorflow.keras.layers import Embedding, Dense, Concatenate, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, \
    Input
import matplotlib.pyplot as plt


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

# 嵌入矩阵的维度
embed_dim = 32
# 用户ID个数
uid_max = max(features.take(0, 1)) + 1  # 6040
# 性别个数
gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2
# 年龄类别个数
age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7
# 职业个数
job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21

# 电影ID个数
movie_id_max = max(features.take(1, 1)) + 1  # 3952
# 电影类型个数
movie_categories_max = max(genres2int.values()) + 1  # 18 + 1 = 19
# 电影名单词个数
movie_title_max = len(title_set)  # 5216

# 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

# 电影名长度
sentences_size = title_count  # = 15
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
# 文本卷积核数量
filter_num = 8

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './save'


def create_model():
    # 获取模型输入数据
    uid = Input(shape=(1,), dtype=tf.int32)
    user_gender = Input(shape=(1,), dtype=tf.int32)
    user_age = Input(shape=(1,), dtype=tf.int32)
    user_job = Input(shape=(1,), dtype=tf.int32)

    movie_id = Input(shape=(1,), dtype=tf.int32)
    movie_categories = Input(shape=(18,), dtype=tf.int32)
    movie_titles = Input(shape=(15,), dtype=tf.int32)

    # 用户特征embedding
    uid_embed_layer = tf.keras.layers.Embedding(uid_max, embed_dim, input_length=1)(uid)
    gender_embed_layer = tf.keras.layers.Embedding(gender_max, embed_dim // 2, input_length=1)(user_gender)
    age_embed_layer = tf.keras.layers.Embedding(age_max, embed_dim // 2, input_length=1)(user_age)
    job_embed_layer = tf.keras.layers.Embedding(job_max, embed_dim // 2, input_length=1)(user_job)

    # 第一层全连接
    uid_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(uid_embed_layer)
    gender_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(gender_embed_layer)
    age_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(age_embed_layer)
    job_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(job_embed_layer)

    # 第二层全连接
    user_combine_layer = tf.keras.layers.concatenate([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer],
                                                     2)  # (?, 1, 128)
    user_combine_layer = tf.keras.layers.Dense(200, activation='tanh')(user_combine_layer)  # (?, 1, 200)

    user_combine_layer_flat = tf.keras.layers.Reshape([200])(user_combine_layer)

    # 电影特征的embedding
    movie_id_embed_layer = tf.keras.layers.Embedding(movie_id_max, embed_dim, input_length=1)(movie_id)

    movie_categories_embed_layer = tf.keras.layers.Embedding(movie_categories_max, embed_dim, input_length=18)(
        movie_categories)
    movie_categories_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(
        movie_categories_embed_layer)

    # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    movie_title_embed_layer = tf.keras.layers.Embedding(movie_title_max, embed_dim, input_length=15)(movie_titles)
    sp = movie_title_embed_layer.shape
    movie_title_embed_layer_expand = tf.keras.layers.Reshape([sp[1], sp[2], 1])(movie_title_embed_layer)
    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        conv_layer = tf.keras.layers.Conv2D(filter_num, (window_size, embed_dim), 1, activation='relu')(
            movie_title_embed_layer_expand)
        maxpool_layer = tf.keras.layers.MaxPooling2D(pool_size=(sentences_size - window_size + 1, 1), strides=1)(
            conv_layer)
        pool_layer_lst.append(maxpool_layer)
    # Dropout层
    pool_layer = tf.keras.layers.concatenate(pool_layer_lst, 3)
    max_num = len(window_sizes) * filter_num
    pool_layer_flat = tf.keras.layers.Reshape([1, max_num])(pool_layer)

    dropout_layer = tf.keras.layers.Dropout(dropout_keep)(pool_layer_flat)

    # 第一层全连接
    movie_id_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(movie_id_embed_layer)
    movie_categories_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(movie_categories_embed_layer)

    # 第二层全连接
    movie_combine_layer = tf.keras.layers.concatenate([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)
    movie_combine_layer = tf.keras.layers.Dense(200, activation='tanh')(movie_combine_layer)

    movie_combine_layer_flat = tf.keras.layers.Reshape([200])(movie_combine_layer)

    # 将用户特征和电影特征作为输入，经过全连接，输出一个预测值
    inference_layer = keras.layers.concatenate([user_combine_layer_flat, movie_combine_layer_flat], 1)
    inference_dense = Dense(64, kernel_regularizer=tf.nn.l2_loss, activation='relu')(inference_layer)
    inference = Dense(1, activation='relu')(inference_dense)

    model = keras.Model(inputs=[uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles],
                        outputs=[inference])
    return model


model = create_model()
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate)

model.summary()

# @tf.function  # 将训练过程转化为图执行模式 运行速度更快
def train(x, y):
    with tf.GradientTape() as tape:
        logits = model([x[0], x[1], x[2], x[3], x[4], x[5], x[6]], training=True)
        loss = loss_fn(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


losses = {'train': [], 'test': []}

# 在整个数据集中训练5轮
for epoch_i in range(num_epochs):

    print(f'epoch_i:{epoch_i}')

    train_X, test_X, train_y, test_y = train_test_split(features, targets_values, test_size=0.2, random_state=0)

    # train
    train_batches = get_batches(train_X, train_y, batch_size)
    batch_num = len(train_X) // batch_size

    for batch_i in range(batch_num):
        x, y = next(train_batches)
        categories = np.zeros([batch_size, 18])
        for i in range(batch_size):
            categories[i] = x.take(6, 1)[i]

        titles = np.zeros([batch_size, sentences_size])
        for i in range(batch_size):
            titles[i] = x.take(5, 1)[i]
        temp=x.take(0,1)
        print(temp)
        loss = train([np.reshape(x.take(0, 1), [batch_size, 1]).astype(np.float32),
                      np.reshape(x.take(2, 1), [batch_size, 1]).astype(np.float32),
                      np.reshape(x.take(3, 1), [batch_size, 1]).astype(np.float32),
                      np.reshape(x.take(4, 1), [batch_size, 1]).astype(np.float32),
                      np.reshape(x.take(1, 1), [batch_size, 1]).astype(np.float32), categories.astype(np.float32),
                      titles.astype(np.float32)], np.reshape(y, [batch_size, 1]).astype(np.float32))

        losses['train'].append(loss)
        print(f'epoch_i:{epoch_i} batch_i:{batch_i} train_loss:{loss}')

    # test
    test_batches = get_batches(test_X, test_y, batch_size)
    batch_num = len(test_X) // batch_size

    for batch_i in range(batch_num):
        x, y = next(test_batches)
        categories = np.zeros([batch_size, 18])
        for i in range(batch_size):
            categories[i] = x.take(6, 1)[i]

        titles = np.zeros([batch_size, sentences_size])
        for i in range(batch_size):
            titles[i] = x.take(5, 1)[i]

        logits = model([np.reshape(x.take(0, 1), [batch_size, 1]).astype(np.float32),
                        np.reshape(x.take(2, 1), [batch_size, 1]).astype(np.float32),
                        np.reshape(x.take(3, 1), [batch_size, 1]).astype(np.float32),
                        np.reshape(x.take(4, 1), [batch_size, 1]).astype(np.float32),
                        np.reshape(x.take(1, 1), [batch_size, 1]).astype(np.float32),
                        categories.astype(np.float32),
                        titles.astype(np.float32)], training=False)
        test_loss = loss_fn(np.reshape(y, [batch_size, 1]).astype(np.float32), logits)
        losses['test'].append(test_loss)
        print(f'epoch_i:{epoch_i} batch_i:{batch_i} test_loss:{test_loss}')



plt.plot(losses['train'], label='Training loss')
plt.legend()
plt.show()

plt.plot(losses['test'], label='Test loss')
plt.legend()
plt.show()
