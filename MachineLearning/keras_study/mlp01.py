#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # , accuracy_score

"""
多层感知机 处理逻辑回归
"""


# 二分类数据
def target_data(num):
    score = np.random.randint(40, 150, size=(num, 2))
    y = np.zeros([num, 1])
    for i in range(num):
        item0 = score[i, 0]
        item1 = score[i, 1]
        # 满足这个条件 y = 1
        if (item0 + item1) > 200 and (item0 > 80) and (item1 > 80):
            y[i, 0] = 1

    return score, y


def test01():
    X, y = target_data(10000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=30
    )

    # 建立一个 Sequential 顺序模型
    model = Sequential()

    # 通过 add() 叠加各层网络
    model.add(Dense(units=64, input_dim=2, activation="sigmoid"))
    model.add(Dense(units=1, activation="sigmoid"))

    # 查看模型结构
    model.summary()

    # 通过 compile() 配置模型求解过程参数
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 训练模型
    model.fit(X_train, y_train, epochs=10, verbose=True)

    # 训练数据集评估
    y_train_predict = model.predict(X_train)
    r2 = r2_score(y_train, y_train_predict)
    print("r2_score:", r2)
    # accuracy = accuracy_score(y_train, y_train_predict)
    # print("accuracy:", accuracy)
    mean_squared = mean_squared_error(y_train, y_train_predict)
    print("mean_squared:", mean_squared)

    # 测试数据集评估
    y_test_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_test_predict)
    print("r2_score:", r2)
    # accuracy = accuracy_score(y_test, y_test_predict)
    # print("accuracy:", accuracy)
    mean_squared = mean_squared_error(y_test, y_test_predict)
    print("mean_squared:", mean_squared)

    # 测试一个数据
    x_test = np.array([[98, 103]])
    y_test = model.predict(x_test)
    print(f"x_test:{x_test},y_test:{y_test}")
    x_test = np.array([[104, 103]])
    y_test = model.predict(x_test)
    print(f"x_test:{x_test},y_test:{y_test}")
    x_test = np.array([[60, 79]])
    y_test = model.predict(x_test)
    print(f"x_test:{x_test},y_test:{y_test}")


if __name__ == "__main__":
    test01()
