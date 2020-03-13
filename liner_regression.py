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
import matplotlib.pyplot as plt

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
    global maximum
    global minimum
    global avgs
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

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w -= gradient_w * eta
        self.b -= gradient_b * eta

    def train(self, train_data, iter=1000, eta=0.01):
        losses = []
        x = train_data[:, :-1]
        y = train_data[:, -1:]
        for i in range(iter):
            z = self.calc(x)
            L = self.loss(z, y)
            gradient_1, gradient_2 = self.gradient(x, y)
            self.update(gradient_1, gradient_2, eta)
            losses.append(L)
        return losses

    def train_sgd(self, train_data, iter=100, batch_size=100, eta=0.01):
        losses = []
        np.random.shuffle(train_data)
        for i in range(iter):
            mini_batches = [train_data[j:j+batch_size] for j in range(0, len(train_data), batch_size)]
            for mini_batch in mini_batches:
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                z = self.calc(x)
                L = self.loss(z, y)
                gradient_1, gradient_2 = self.gradient(x, y)
                self.update(gradient_1, gradient_2, eta)
                losses.append(L)
        return losses

    def predict(self, x):
        z = self.calc(x)
        return z * (maximum[-1] - minimum[-1]) + avgs[-1]
        # return z


if __name__ == '__main__':
    ori_data = load_data(data_dir, 'housing.data')
    o_data = load_data(data_dir, 'housing.data')
    train_data, test_data = process_data(ori_data)
    network = LinerRegression(13)
    plot_y = network.train_sgd(train_data, iter=200)
    plot_x = range(len(plot_y))
    plt.plot(plot_x, plot_y)
    plt.show()


