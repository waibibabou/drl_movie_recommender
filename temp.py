import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from collections import deque
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
import random


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
    ratings = ratings.filter(regex='UserID|MovieID|ratings|timestamps')

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    # 按照userid 以及 timestamps排序
    data.sort_values(by=['UserID', 'timestamps'], axis=0, ascending=[True, True], inplace=True)
    # 之后将timestamps去除
    data = data.drop(['timestamps'], axis=1)

    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets_values = targets_pd.values

    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig


title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()

# 嵌入矩阵的维度
embed_dim = 32
# 用户ID个数 用户id从1开始
uid_max = max(features.take(0, 1)) + 1
# 性别个数
gender_max = max(features.take(2, 1)) + 1
# 年龄类别个数
age_max = max(features.take(3, 1)) + 1
# 职业个数
job_max = max(features.take(4, 1)) + 1

# 电影ID个数 电影id从1开始
movie_id_max = max(features.take(1, 1)) + 1
# 电影类型个数
movie_categories_max = max(genres2int.values()) + 1
# 电影名单词个数
movie_title_max = len(title_set)

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
    # 状态中加入用户兴趣向量
    user_interest = Input(shape=(movie_categories_max,), dtype=tf.float32)

    movie_id = Input(shape=(1,), dtype=tf.int32)
    movie_categories = Input(shape=(18,), dtype=tf.int32)
    movie_titles = Input(shape=(15,), dtype=tf.int32)

    # 用户特征embedding
    uid_embed_layer = tf.keras.layers.Embedding(uid_max, embed_dim, input_length=1)(uid)
    gender_embed_layer = tf.keras.layers.Embedding(gender_max, embed_dim // 2, input_length=1)(user_gender)
    age_embed_layer = tf.keras.layers.Embedding(age_max, embed_dim // 2, input_length=1)(user_age)
    job_embed_layer = tf.keras.layers.Embedding(job_max, embed_dim // 2, input_length=1)(user_job)
    # userinterest各个位置最大为1
    uinterest_embed_layer = tf.keras.layers.Embedding(2, embed_dim // 2, input_length=movie_categories_max)(
        user_interest)
    uinterest_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(
        uinterest_embed_layer)

    # 第一层全连接
    uid_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(uid_embed_layer)
    gender_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(gender_embed_layer)
    age_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(age_embed_layer)
    job_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(job_embed_layer)
    # uinterest_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(user_interest)
    uinterest_fc_layer = tf.keras.layers.Dense(embed_dim, activation='relu')(uinterest_embed_layer)

    # 第二层全连接
    user_combine_layer = tf.keras.layers.concatenate(
        [uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer, uinterest_fc_layer], 2)  # (?, 1, 160)
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

    # 将用户特征和电影特征作为输入，经过全连接，输出5个评分的Q值
    inference_layer = keras.layers.concatenate([user_combine_layer_flat, movie_combine_layer_flat], 1)
    inference_dense = Dense(64, kernel_regularizer=tf.nn.l2_loss, activation='relu')(inference_layer)

    inference = Dense(5)(inference_dense)

    model = keras.Model(
        inputs=[uid, user_gender, user_age, user_job, user_interest, movie_id, movie_categories, movie_titles],
        outputs=[inference])
    return model


# 当前Q网络
current_net = create_model()
# 目标Q网络
target_net = create_model()
# 两网络初始参数相同
current_net.save('current_net_weights1.h5')
target_net.load_weights('current_net_weights1.h5')


# loss_fn = keras.losses.MeanSquaredError()
# optimizer = keras.optimizers.Adam(learning_rate)
#
#
# # @tf.function  # 将训练过程转化为图执行模式 运行速度更快
# def train(x, y):
#     with tf.GradientTape() as tape:
#         logits = model([x[0], x[1], x[2], x[3], x[4], x[5], x[6]], training=True)
#         loss = loss_fn(y, logits)
#
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return loss
#
#
# def get_batches(Xs, ys, batch_size):
#     for start in range(0, len(Xs), batch_size):
#         end = min(start + batch_size, len(Xs))
#         yield Xs[start:end], ys[start:end]
#
#
# losses = {'train': [], 'test': []}
#
# # 在整个数据集中训练5轮
# for epoch_i in range(num_epochs):
#
#     print(f'epoch_i:{epoch_i}')
#
#     train_X, test_X, train_y, test_y = train_test_split(features, targets_values, test_size=0.2, random_state=0)
#
#     # train
#     train_batches = get_batches(train_X, train_y, batch_size)
#     batch_num = len(train_X) // batch_size
#
#     for batch_i in range(batch_num):
#         x, y = next(train_batches)
#         categories = np.zeros([batch_size, 18])
#         for i in range(batch_size):
#             categories[i] = x.take(6, 1)[i]
#
#         titles = np.zeros([batch_size, sentences_size])
#         for i in range(batch_size):
#             titles[i] = x.take(5, 1)[i]
#         temp=x.take(0,1)
#         print(temp)
#         loss = train([np.reshape(x.take(0, 1), [batch_size, 1]).astype(np.float32),
#                       np.reshape(x.take(2, 1), [batch_size, 1]).astype(np.float32),
#                       np.reshape(x.take(3, 1), [batch_size, 1]).astype(np.float32),
#                       np.reshape(x.take(4, 1), [batch_size, 1]).astype(np.float32),
#                       np.reshape(x.take(1, 1), [batch_size, 1]).astype(np.float32), categories.astype(np.float32),
#                       titles.astype(np.float32)], np.reshape(y, [batch_size, 1]).astype(np.float32))
#
#         losses['train'].append(loss)
#         print(f'epoch_i:{epoch_i} batch_i:{batch_i} train_loss:{loss}')
#
#     # test
#     test_batches = get_batches(test_X, test_y, batch_size)
#     batch_num = len(test_X) // batch_size
#
#     for batch_i in range(batch_num):
#         x, y = next(test_batches)
#         categories = np.zeros([batch_size, 18])
#         for i in range(batch_size):
#             categories[i] = x.take(6, 1)[i]
#
#         titles = np.zeros([batch_size, sentences_size])
#         for i in range(batch_size):
#             titles[i] = x.take(5, 1)[i]
#
#         logits = model([np.reshape(x.take(0, 1), [batch_size, 1]).astype(np.float32),
#                         np.reshape(x.take(2, 1), [batch_size, 1]).astype(np.float32),
#                         np.reshape(x.take(3, 1), [batch_size, 1]).astype(np.float32),
#                         np.reshape(x.take(4, 1), [batch_size, 1]).astype(np.float32),
#                         np.reshape(x.take(1, 1), [batch_size, 1]).astype(np.float32),
#                         categories.astype(np.float32),
#                         titles.astype(np.float32)], training=False)
#         test_loss = loss_fn(np.reshape(y, [batch_size, 1]).astype(np.float32), logits)
#         losses['test'].append(test_loss)
#         print(f'epoch_i:{epoch_i} batch_i:{batch_i} test_loss:{test_loss}')
#
#
#
# plt.plot(losses['train'], label='Training loss')
# plt.legend()
# plt.show()
#
# plt.plot(losses['test'], label='Test loss')
# plt.legend()
# plt.show()

def select_action_and_change_e(ll) -> int:
    global current_epsilon
    temp = random.random()
    ans = 0
    if temp > current_epsilon:
        ans = np.argmax(ll)
    else:
        step = int(temp / (current_epsilon / 4))
        if step >= np.argmax(ll):
            step += 1
        ans = step

    current_epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 240000
    return ans + 1


@tf.function
def train(x, y, action):
    with tf.GradientTape() as tape:
        logits = current_net([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]], training=True)
        l = []
        for i in range(BATCH_SIZE):
            l.append(logits[i][action[i] - 1])
        l = tf.convert_to_tensor(l, dtype=tf.float32)

        # logits = logits[0][action - 1]
        # logits = tf.reshape(logits, [1, 1])
        # logits=tf.reshape(logits,[BATCH_SIZE,1])
        loss = loss_fn(y, l)

    grads = tape.gradient(loss, current_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, current_net.trainable_variables))
    return loss


# DQN模型的一些超参数
GAMMA = 0.9  # 折扣因子
INITIAL_EPSILON = 0.5  # epsilon的最初值
FINAL_EPSILON = 0.01  # epsilon的最终值
REPLAY_SIZE = 10000  # 经验回放池中的样本最大数量
BATCH_SIZE = 32  # 训练时每次从经验回放池中取出的样本数量
REPLACE_TARGET_FREQ = 10  # 更新目标网络的频率

# loss函数以及训练方法
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate)

# 经验回放池
replay_buffer = deque()

# 分出训练集测试集 测试集占20%
train_X, test_X = features[:80000], features[-20000:]
train_Y, test_Y = targets_values[:80000], targets_values[-20000:]

# 用户的兴趣向量
user_interest = [0] * movie_categories_max

# 设定epsilon
current_epsilon = INITIAL_EPSILON
# 设定reward映射
reward_list = [1, 0.5, 0, -0.5, -1]
# 改变兴趣向量
change_interest_list = [-0.1, -0.05, +0.05, +0.1, +0.15]
# 记录训练次数
train_step = 0
# 记录训练loss以及测试loss用于画图
losses = {'train': [], 'test': []}

for epoch_i in range(num_epochs):
    print(f'\n\nepoch:{epoch_i} train begin')

    for i in range(len(train_X)):

        if (i + 1) % 20000 == 0:

            # 在测试集中测试
            print(f'\n\nepoch:{epoch_i} test begin')

            temp_loss = 0

            for ii in range(len(test_X)):
                if ii % BATCH_SIZE == 0:
                    temp_loss = 0

                # 换人的时候需要重新随机初始化用户的兴趣向量
                if ii == 0 or test_X[ii][0] != test_X[ii - 1][0]:
                    _user_interest = [random.random() for _ in range(movie_categories_max)]

                # 每次输入一条数据
                x, y = test_X[ii], test_Y[ii]
                # 将用户向量加到状态最后的位置
                x = x.tolist()
                x.append(_user_interest)
                x = np.array(x)

                x = np.reshape(x, [1, len(x)])
                y = np.reshape(y, [1, 1])

                categories = np.zeros([1, 18])
                categories[0] = x.take(6, 1)[0]
                titles = np.zeros([1, sentences_size])
                titles[0] = x.take(5, 1)[0]
                interest = np.zeros([1, movie_categories_max])
                interest[0] = x.take(7, 1)[0]

                # 输入到当前Q网络得到所有Q值，选取最大的Q值对应的动作
                all_actions_Q = current_net([np.reshape(x.take(0, 1), [1, 1]).astype(np.float32),
                                             np.reshape(x.take(2, 1), [1, 1]).astype(np.float32),
                                             np.reshape(x.take(3, 1), [1, 1]).astype(np.float32),
                                             np.reshape(x.take(4, 1), [1, 1]).astype(np.float32),
                                             interest.astype(np.float32),
                                             np.reshape(x.take(1, 1), [1, 1]).astype(np.float32),
                                             categories.astype(np.float32),
                                             titles.astype(np.float32)], training=False)
                all_actions_Q = np.reshape(all_actions_Q, [5])

                # 根据贪婪法选择动作
                action = np.argmax(all_actions_Q) + 1

                # 更新用户兴趣向量
                invalid_index = genres2int['<PAD>']
                for j in range(18):
                    if categories[0][j] != invalid_index:
                        _user_interest[int(categories[0][j])] += change_interest_list[action - 1]
                        # 用户兴趣向量各个位置最大值为1 最小值为0
                        if _user_interest[int(categories[0][j])] > 1:
                            _user_interest[int(categories[0][j])] = 1
                        elif _user_interest[int(categories[0][j])] < 0:
                            _user_interest[int(categories[0][j])] = 0
                    else:
                        break

                temp_loss += (action - y[0][0]) ** 2
                if (ii + 1) % BATCH_SIZE == 0:
                    losses['test'].append((temp_loss / BATCH_SIZE) ** 0.5)
                    print(
                        f'epoch: {epoch_i} batch: {int((ii + 1) / BATCH_SIZE)} test loss: {(temp_loss / BATCH_SIZE) ** 0.5}')

        # 换人的时候需要随机初始化用户的兴趣向量
        if i == 0 or train_X[i][0] != train_X[i - 1][0]:
            user_interest = [random.random() for _ in range(movie_categories_max)]

        # 每次输入一条数据
        x, y = train_X[i], train_Y[i]
        # 将用户兴趣向量加到状态最后的位置
        x = x.tolist()
        x.append(user_interest)
        x = np.array(x)
        # 改变维度
        x = np.reshape(x, [1, len(x)])
        y = np.reshape(y, [1, 1])

        categories = np.zeros([1, 18])
        categories[0] = x.take(6, 1)[0]
        titles = np.zeros([1, sentences_size])
        titles[0] = x.take(5, 1)[0]
        interest = np.zeros([1, movie_categories_max])
        interest[0] = x.take(7, 1)[0]

        # 输入到当前Q网络中得到所有动作的Q值
        all_actions_Q = current_net([np.reshape(x.take(0, 1), [1, 1]).astype(np.float32),
                                     np.reshape(x.take(2, 1), [1, 1]).astype(np.float32),
                                     np.reshape(x.take(3, 1), [1, 1]).astype(np.float32),
                                     np.reshape(x.take(4, 1), [1, 1]).astype(np.float32),
                                     interest.astype(np.float32),
                                     np.reshape(x.take(1, 1), [1, 1]).astype(np.float32),
                                     categories.astype(np.float32),
                                     titles.astype(np.float32)], training=False)
        all_actions_Q = np.reshape(all_actions_Q, [5])

        # 根据epsilon贪婪法选择动作
        action = select_action_and_change_e(all_actions_Q)
        # 根据预测评分与实际评分的差值得到奖励
        reward = reward_list[abs(action - y[0][0])]
        # 更新用户兴趣向量
        user_interest_ = user_interest.copy()
        invalid_index = genres2int['<PAD>']
        for j in range(18):
            if categories[0][j] != invalid_index:
                user_interest_[int(categories[0][j])] += change_interest_list[action - 1]
                # 用户兴趣向量各个位置最大值为1 最小值为0
                if user_interest_[int(categories[0][j])] > 1:
                    user_interest_[int(categories[0][j])] = 1
                elif user_interest_[int(categories[0][j])] < 0:
                    user_interest_[int(categories[0][j])] = 0
            else:
                break

        # 将五元组存入经验回放池中(当前状态、动作、奖励、下一状态、是否为终点)
        S = x
        if i == len(train_X) - 1 or train_X[i + 1][0] != train_X[i][0]:
            end = True
        else:
            end = False
        if not end:
            S_next = train_X[i + 1]
            S_next = S_next.tolist()
            S_next.append(user_interest_)
            S_next = np.array(S_next)
        else:
            S_next = []

        if len(replay_buffer) >= REPLAY_SIZE:
            replay_buffer.popleft()
        replay_buffer.append([S, action, reward, S_next, end])
        # 更新兴趣向量
        user_interest = user_interest_

        # 经验回放池中的样本数量已经足够，可以训练当前Q网络
        if len(replay_buffer) >= BATCH_SIZE and i % 100 == 0:  # （测试：：减少训练时间）
            samples = random.sample(replay_buffer, BATCH_SIZE)
            # 用于存储目标值
            y = [0] * BATCH_SIZE
            # 计算这个batch中所有y值
            for index in range(BATCH_SIZE):
                # 该样本为最后的
                if samples[index][4]:
                    y[index] = samples[index][2]
                else:
                    S_next = samples[index][3]
                    # 改变维度
                    S_next = np.reshape(S_next, [1, len(S_next)])

                    categories = np.zeros([1, 18])
                    categories[0] = S_next.take(6, 1)[0]
                    titles = np.zeros([1, sentences_size])
                    titles[0] = S_next.take(5, 1)[0]
                    interest = np.zeros([1, movie_categories_max])
                    interest[0] = S_next.take(7, 1)[0]

                    Q_list = target_net([np.reshape(S_next.take(0, 1), [1, 1]).astype(np.float32),
                                         np.reshape(S_next.take(2, 1), [1, 1]).astype(np.float32),
                                         np.reshape(S_next.take(3, 1), [1, 1]).astype(np.float32),
                                         np.reshape(S_next.take(4, 1), [1, 1]).astype(np.float32),
                                         interest.astype(np.float32),
                                         np.reshape(S_next.take(1, 1), [1, 1]).astype(np.float32),
                                         categories.astype(np.float32),
                                         titles.astype(np.float32)],
                                        training=False)
                    Q_list = np.reshape(Q_list, [5])
                    y[index] = samples[index][2] + GAMMA * max(Q_list)

            # 得到所有y值后计算Q值并反向传播训练当前Q网络 训练时一个一个送入
            # loss_temp = 0
            # for index in range(BATCH_SIZE):
            #     S_temp=samples[index][0]#S_temp已经为二维
            #
            #     categories = np.zeros([1, 18])
            #     categories[0] = S_temp.take(6,1)[0]
            #     titles = np.zeros([1, sentences_size])
            #     titles[0] = S_temp.take(5,1)[0]
            #     interest = np.zeros([1, movie_categories_max])
            #     interest[0] = S_temp.take(7,1)[0]
            #     loss = train([np.reshape(S_temp.take(0,1), [1, 1]).astype(np.float32),
            #                   np.reshape(S_temp.take(2,1), [1, 1]).astype(np.float32),
            #                   np.reshape(S_temp.take(3,1), [1, 1]).astype(np.float32),
            #                   np.reshape(S_temp.take(4,1), [1, 1]).astype(np.float32),
            #                   interest.astype(np.float32),
            #                   np.reshape(S_temp.take(1,1), [1, 1]).astype(np.float32),
            #                   categories.astype(np.float32),
            #                   titles.astype(np.float32)],
            #                   y=np.reshape(y[index], [1, 1]).astype(np.float32),
            #                   action=samples[index][1])
            #     loss_temp += loss.numpy()

            S_list = []
            action_list = []
            for index in range(BATCH_SIZE):
                S_list.append(samples[index][0][0])
                action_list.append(samples[index][1])
            S_list = np.array(S_list)
            action_list = np.array(action_list)

            categories = np.zeros([BATCH_SIZE, 18])
            titles = np.zeros([BATCH_SIZE, sentences_size])
            interest = np.zeros([BATCH_SIZE, movie_categories_max])
            for index in range(BATCH_SIZE):
                categories[index] = S_list.take(6, 1)[index]
                titles[index] = S_list.take(5, 1)[index]
                interest[index] = S_list.take(7, 1)[index]
            # 将这批数据送入当前网络中训练
            loss = train([np.reshape(S_list.take(0, 1), [BATCH_SIZE, 1]).astype(np.float32),
                          np.reshape(S_list.take(2, 1), [BATCH_SIZE, 1]).astype(np.float32),
                          np.reshape(S_list.take(3, 1), [BATCH_SIZE, 1]).astype(np.float32),
                          np.reshape(S_list.take(4, 1), [BATCH_SIZE, 1]).astype(np.float32),
                          interest.astype(np.float32),
                          np.reshape(S_list.take(1, 1), [BATCH_SIZE, 1]).astype(np.float32),
                          categories.astype(np.float32),
                          titles.astype(np.float32)],
                         y=np.reshape(y, [BATCH_SIZE, 1]).astype(np.float32),
                         action=action_list)

            print(f'epoch: {epoch_i} batch: {i} train loss: {loss}')
            losses['train'].append(loss)

            train_step += 1
            if train_step % REPLACE_TARGET_FREQ == 0:
                # 更新目标网络参数为当前网络参数
                current_net.save('current_net_weights1.h5')
                target_net.load_weights('current_net_weights1.h5')

# 画图
plt.plot(losses['train'], label='train_loss')
plt.xlabel('steps')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

plt.plot(losses['test'], label='test_loss')
plt.xlabel('steps')
plt.ylabel('RMSE Loss')
plt.legend()
plt.show()
