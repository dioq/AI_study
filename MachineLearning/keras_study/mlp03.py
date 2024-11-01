#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

"""
多层感知机 拟合多项式函数
"""


def test01():
    x = np.linspace(-100, 100, 10000)
    y = 1 + x + 2 * x**2 + 3 * x**3  # + 4 * x**4
    X_train, X_test, y_train, y_test = train_test_split(
        x.reshape(-1, 1), y.reshape(-1, 1), test_size=0.25, random_state=30
    )

    # 建立一个 Sequential 顺序模型
    model = Sequential()
    model.add(Dense(units=64, activation="relu", input_dim=1))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=1, activation="linear"))
    # 编译模型，选择优化器和损失函数
    model.compile(optimizer="adam", loss="mean_squared_error")

    # 训练模型
    model.fit(X_train, y_train, epochs=1000)

    model.summary()  # 打印模型概述信息

    # 训练数据集评估
    y_train_predict = model.predict(X_train)
    r2 = r2_score(y_train, y_train_predict)
    print("r2_score:", r2)
    mean_squared = mean_squared_error(y_train, y_train_predict)
    print("mean_squared:", mean_squared)

    # 测试数据集评估
    y_test_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_test_predict)
    print("r2_score:", r2)
    mean_squared = mean_squared_error(y_test, y_test_predict)
    print("mean_squared:", mean_squared)

    # 可视化
    plt.figure(figsize=(16, 9))
    plt.title("Title")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, color="red", marker="*", label="origin")
    plt.scatter(X_test, y_test_predict, color="blue", marker="+", label="test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test01()
