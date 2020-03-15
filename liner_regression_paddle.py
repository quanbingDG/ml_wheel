#!/bin/bash/env python
# -*- coding:utf8 -*-

"""
@author:quanbing
@project:ml_wheel
@file:liner_regression_paddle.py
@time:2020/03/13
"""


from paddle.fluid.dygraph import Linear
import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph.base import to_variable


def process_data(ratio=0.8):
    housing = np.fromfile(r'D:\UD\workspace\ml_wheel\data\housing.data', sep=' ')
    # 定义特征列名
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                          'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)
    # 改变数据结构
    housing = housing.reshape(housing.shape[0] // feature_num, feature_num)
    # 拆分训练集、测试集
    offset = int(housing.shape[0] * ratio)
    train_data = housing[:offset]
    test_data = housing[offset:]
    # 归一化处理
    maximum, minimum, avgs = train_data.max(axis=0), train_data.min(axis=0), \
                             train_data.mean(axis=0)
    global MA
    global MI
    global AV
    MA, MI, AV = maximum, minimum, avgs
    for i in range(feature_num):
        train_data[:, i] = (train_data[:, i] - avgs[i]) / (maximum[i] - minimum[i])
        test_data[:, i] = (test_data[:, i] - avgs[i]) / (maximum[i] - minimum[i])
    return train_data, test_data


class NetWork(fluid.dygraph.Layer):
    def __init__(self, x_num):
        super(NetWork, self).__init__()
        self.liner = Linear(x_num, 1, bias_attr=True, dtype='double')

    def forword(self, x):
        return self.liner(x)


if __name__ == '__main__':
    # 加载数据
    train_data, test_data = process_data()
    # 转换成paddle格式
    with fluid.dygraph.guard():
        training_data = train_data[:, :-1]
        # training_data = to_variable(training_data)
        model = NetWork(13)
        # print(model.forword(training_data))
        model.train()
    # 定义目标函数、优化算法
    place = fluid.CPUPlace()
    main = fluid.Program()
    with fluid.program_guard(main):
        opt = fluid.optimizer.SGD(learning_rate=0.01)
        EPOCH_NUM = 10  # 设置外层循环次数
        BATCH_SIZE = 10  # 设置batch大小

        # 定义外层循环
        for epoch_id in range(EPOCH_NUM):
            # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个batch包含10条数据
            mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
            # 定义内层循环
            for iter_id, mini_batch in enumerate(mini_batches):
                x = np.array(mini_batch[:, :-1]).astype('float32')  # 获得当前批次训练数据
                y = np.array(mini_batch[:, -1:]).astype('float32')  # 获得当前批次训练标签（真实房价）
                # 将numpy数据转为飞桨动态图variable形式
                house_features = dygraph.to_variable(x)
                prices = dygraph.to_variable(y)

                # 前向计算
                predicts = model(house_features)

                # 计算损失
                loss = fluid.layers.square_error_cost(predicts, label=prices)
                avg_loss = fluid.layers.mean(fluid.layers.sqrt(loss))
                if iter_id % 20 == 0:
                    print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

                # 反向传播
                avg_loss.backward()
                # 最小化loss,更新参数
                opt.minimize(avg_loss)
                # 清除梯度
                model.clear_gradients()
        # 保存模型
        fluid.save_dygraph(model.state_dict(), 'LR_model')



