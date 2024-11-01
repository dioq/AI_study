#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

"""
sklearn.svm.SVR(
    kernel ='rbf',
    degree = 3,
    gamma ='auto_deprecated',
    coef0 = 0.0,
    tol = 0.001,
    C = 1.0,
    epsilon = 0.1,
    shrinking = True,
    cache_size = 200,
    verbose = False,
    max_iter = -1)
使用支持向量和相应的核函数创建一个超平面,以最小化预测误差和复杂度. 常使用凸优化技术来求解,尽量保证找到全局最优解

参数:	
kernel:         string,optional(default ='rbf')
指定要在算法中使用的内核类型。它必须是'linear','poly','rbf','sigmoid','precomputed'或者callable之一。
如果没有给出,将使用'rbf'。如果给出了callable,则它用于预先计算内核矩阵

degree: int,可选(默认= 3)
多项式核函数的次数('poly')。被所有其他内核忽略

gamma:float,optional(默认='auto')
'rbf','poly'和'sigmoid'的核系数
当前默认值为'auto',它使用1 / n_features,如果gamma='scale'传递,则使用1 /(n_features * X.std())作为gamma的值。当前默认的gamma''auto'将在版本0.22中更改为'scale'。'auto_deprecated','auto'的弃用版本用作默认值,表示没有传递明确的gamma值。

coef0:float,optional(默认值= 0.0)
核函数中的独立项。它只在'poly'和'sigmoid'中很重要。

tol:float,optional(默认值= 1e-3)
容忍停止标准。

C:float,可选(默认= 1.0)
错误术语的惩罚参数C.

epsilon:float,optional(默认值= 0.1)
Epsilon在epsilon-SVR模型中。它指定了epsilon-tube,其中训练损失函数中没有惩罚与在实际值的距离epsilon内预测的点。
收缩:布尔值,可选(默认= True)是否使用收缩启发式。

cache_size:float,可选
指定内核缓存的大小(以MB为单位)。

max_iter:int,optional(默认值= -1)
求解器内迭代的硬限制,或无限制的-1


在绝大多数的非线性模型中(包括一些特殊的线性模型,比如ridge regression),还有一部分参数是无法通过训练直接获取的,
通常的做法是直接预先设定,反复调整使模型达到一个最优的状态,这一过程也就是我们常听到的"调参"。
"""


def test01():
    x = np.arange(20)
    y = 0.3 * x**3 + 0.4 * x**2 + 0.5 * x + 3

    # 创建支持向量回归模型
    model = svm.SVR(kernel="rbf", gamma=1, C=10000.0)

    # 拟合数据
    x_arrary = x.reshape(-1, 1)
    model.fit(x_arrary, y)

    # 预测新数据
    y_predict = model.predict(x_arrary)

    # 可视化
    plt.figure(figsize=(16, 9))
    plt.title("Support Vector Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, color="blue")
    plt.plot(x, y_predict, color="red", label="predict")
    plt.legend()
    plt.show()


def target_data(column_len):
    score = np.random.randint(40, 150, size=(column_len, 2))
    y = np.zeros(column_len)
    for i in range(column_len):
        item0 = score[i, 0]
        item1 = score[i, 1]
        # 满足这个条件 y = 1
        if (item0 + item1) > 200:  # and (item0 > 80) and (item1 > 80):
            y[i] = 1

    return score, y


def test02():
    X, y = target_data(1000)

    # 训练线性SVM
    # linear_svm = svm.SVR(kernel="linear").fit(X, y)
    # y_predict = linear_svm.predict(X)
    # r2_score_error = r2_score(y, y_predict)
    # print("r2_score:", r2_score_error)
    # squared_error = mean_squared_error(y, y_predict)
    # print("squared_error:", squared_error)
    # 训练带有RBF核的SVM
    rbf_svm = svm.SVR(kernel="rbf", gamma=1, C=10000.0).fit(X, y)
    y_predict = rbf_svm.predict(X)
    r2_score_error = r2_score(y, y_predict)
    print("r2_score:", r2_score_error)
    squared_error = mean_squared_error(y, y_predict)
    print("squared_error:", squared_error)


if __name__ == "__main__":
    # test01()
    test02()
