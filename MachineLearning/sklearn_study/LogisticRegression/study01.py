#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from matplotlib import pyplot as plt

"""
逻辑回归
"""


def target_data(column_len):
    score = np.random.randint(40, 150, size=(column_len, 2))
    y = np.zeros(column_len)
    for i in range(column_len):
        item0 = score[i, 0]
        item1 = score[i, 1]
        # 满足这个条件 y = 1
        if (item0 + item1) > 200 and (item0 > 80) and (item1 > 80):
            y[i] = 1

    return score, y


def test01():
    x, y = target_data(1000)
    # print(x)
    # print(y)

    model = LogisticRegression(C=1.0, penalty="l2", tol=0.01)
    # model = LogisticRegression(
    #     penalty="l2", dual=False, C=2.0, n_jobs=1, random_state=20, fit_intercept=True
    # )
    # 拟合数据
    model.fit(x, y)
    # 获取回归系数
    coef = model.coef_
    print("coef_:", coef)
    # 获取回归截距
    intercept = model.intercept_
    print("intercept_:", intercept)

    # 模型评佑
    y_predict = model.predict(x)
    R2_score = r2_score(y, y_predict)
    print("r2 score:", R2_score)
    mean_squared = mean_squared_error(y, y_predict)
    print("mean squared:", mean_squared)
    accuracy = accuracy_score(y, y_predict)
    print("accuracy score:", accuracy)
    # 函数模型
    # theta0 + theta1 * X1 + theta2 * X2 = 0
    theta0 = intercept
    theta1, theta2 = coef[0][0], coef[0][1]
    X1 = np.arange(40, 150)
    X2 = -(theta0 + theta1 * X1) / theta2
    # print(X1)
    # print("===========")
    # print(X2)

    # 预测
    x_test = [[98, 103]]
    y_test = model.predict(x_test)
    print(f"x_test:{x_test},y_test:{y_test}")

    # 可视化
    plt.figure(figsize=(16, 9))
    plt.title("Logistic Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    mask = y[:] == 1
    plt.scatter(x[:, 0][mask], x[:, 1][mask], color="red", marker="*")
    plt.scatter(x[:, 0][~mask], x[:, 1][~mask], color="blue", marker="+")
    plt.plot(X1, X2, color="green")
    plt.show()


if __name__ == "__main__":
    test01()
