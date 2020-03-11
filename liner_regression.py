#!/bin/bash/env python
# -*- coding:utf8 -*-

"""
@author:quanbing
@project:ml_wheel
@file:liner_regression.py
@time:2020/03/11
"""

##使用numpy手写线性回归并利用梯度下降求解
##波士顿房价预测
#1.数据处理
#2.模型设计
#3.训练配置
#4.训练过程
#5.保存模型

from pathlib import Path
import numpy as np

data_dir = Path(r'D:\UD\workspace\ml_wheel\data')
# 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
feature_num = len(feature_names)


def load_data(file_dir, file_name):
    d = np.fromfile(r'D:\UD\workspace\ml_wheel\data\housing.data', sep=' ')
    d = d.reshape(d.shape[0]//feature_num, feature_num)
    return d


def process_data(data_, train_ratio=0.8):
    # 拆分训练集、测试集
    offset = int(data_.shape[0] * 0.8)
    # 计算最大值、最小值、平均值 用做归一化处理
    maximum, minimum, avgs = data_.max(axis=0), data_.min(axis=0), data_.mean(axis=0)
    # 将属性全部归一化
    for i in range(feature_num):
        data_[:, i] = (data_[:, i] - avgs[i]) / (maximum[i] - minimum[i])
    train = data_[:offset]
    test = data_[offset:]
    return train, test


class LinerRegression:
    def __init__(self, num_of_x):
        '''
        :param num_of_x: X数量，用于初始化权重向量
        '''
        np.random.seed(9527)
        self.w = np.random.randn(num_of_x, 1)
        self.b = 0

    def calc(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    @staticmethod
    def loss(z, y):
        cost = (z - y)**2/2
        return cost.mean()

    def gradient(self, x, y):
        z = self.calc(x)
        gradient_w = (z - y)*x
        gradient_w = gradient_w.mean(axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b


if __name__ == '__main__':
    ori_data = load_data(data_dir, 'housing.data')
    train_data, test_data = process_data(ori_data)
    X = train_data[:, :-1]
    y = train_data[:, -1:]
    network = LinerRegression(13)
    print(X)
    print(X.shape)
    z = network.calc(X)
    print(network.loss(z, y))
    gradient_w, gradient_b = network.gradient(X, y)
    print(gradient_w)
    print(gradient_b)


