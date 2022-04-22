import math
import random
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from typing import *

# a = tf.constant([1, 2, 3], dtype=tf.float32)
# print(a, type(a), a[0])
#
# b = np.array([1, 2, 3], dtype='float32')
# print(type(b))
# b = b.astype('float64')
# b = tf.convert_to_tensor(b)
# print(b, b[0])
# b = b.numpy()
# print(b)
# print('\n\n\n')
#
# c = tf.constant([1, 2, 3], dtype=tf.float32)
# print(c, c.numpy())
# d=tf.cast(c, dtype=tf.int64)
# print(c,d)
#
# # x=np.array([[1,2,3],[4,5,6]])
# # y=np.array([[3,4,5],[8,7,6]])
# # plt.plot(x,y)
# # plt.show()
#
# dicc={1:2,3:4}
# for i in dicc.values():
#     print(i)
#
# print(2 and 2>1)
# print('%.2f %010d %s'%(1.111,2345666,'123333'))
#
# a=tf.constant([[1,2]])
# b=tf.constant([[1],[2]])
# print(tf.matmul(a,b))
# print(a*b)
#
# c=np.mat([1,2])
# print(c,type(c),c.shape)
# c=np.array([1,2])
# print(c,type(c),c.shape)
#
# a=None
# print(a,type(a))
#
# a=[1,2,3]
# print(a[-1])
#
# def a(x:int =None):
#     print(x)
#
# a(1)
# a()
# a=[1,2,3]
# print(a.index(1))
#
# a=[1,2,3]
# print(a[3:])
#
# a={1:[1,2]}
# if 2 not in a:
#     a[2]=[]
# a[2].append(1)
# print(a)


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def __init__(self):
#         self.dicc=dict()
#         self.preorder=[]
#
#     def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
#         index=0
#         count=0
#         for i in range(len(traversal)):
#             if traversal[i].isdigit():
#                 self.preorder.append(traversal[i])
#                 if count not in self.dicc:
#                     self.dicc[count]=[]
#                 self.dicc[count].append(index)
#                 count=0
#                 index+=1
#             else:
#                 count+=1
#         return self.x(0,self.preorder,0,len(self.preorder)-1)
#     def x(self,depth:int,preorder,left:int,right:int)->Optional[TreeNode]:
#         if len(preorder)==0:
#             return None
#         if len(preorder)==1:
#             return TreeNode(val=int(preorder[0]))
#         root=TreeNode(val=int(preorder[0]))
#
#         l=self.dicc[depth+1]
#         leftbegin=left+1
#         rightbegin=0
#         index=l.index(leftbegin)
#         for i in l[index+1:]:
#             if left<=i<=right:
#                 rightbegin=i
#         leftlen=right-leftbegin+1 if rightbegin==0 else rightbegin-leftbegin
#         rightlen=0 if rightbegin==0 else right-rightbegin+1
#         root.left=self.x(depth+1,preorder[1:1+leftlen],leftbegin,leftbegin+leftlen-1)
#         root.right=self.x(depth+1,preorder[1+leftlen:],leftbegin+leftlen,right)
#         return root
#
#
# my=Solution()
# my.recoverFromPreorder("1-401--349---90--88")


# class x:
#     def __init__(self):
#         self.x=10
#
#     def a(self,x):
#         print(x,self.x)
#
# my=x()
# my.a(11)
#
# dicc=dict()
# dicc[1]='1'
# dicc[2]='2'
# print(dicc.keys())
# def xx():
#     global a
#     a+=1
#
# a=10
# xx()
# print(a)
