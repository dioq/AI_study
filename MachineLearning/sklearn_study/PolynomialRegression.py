#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt


def test01():
    # x = np.random.randint(-10, 10, size=(1000, 1))
    X = np.linspace(-100, 100, 100)
    y = 3.3 * X**3 + 2.2 * X**2 + 1.1 * X - 3

    X_train, X_test, y_train, y_test = train_test_split(
        X.reshape(-1, 1), y.reshape(-1, 1), test_size=0.25, random_state=33
    )

    # 创建多项式特征 degree 是多项式最高次数
    poly = PolynomialFeatures(degree=3)
    X_train2 = poly.fit_transform(X_train)
    X_test2 = poly.fit_transform(X_test)

    # 创建线性回归模型并拟合多项式特征
    model = LinearRegression()
    model.fit(X_train2, y_train)
    # 获取回归系数
    print("coef_:", model.coef_)
    # 获取回归截距
    intercept = model.intercept_
    print("intercept_:", intercept)

    print("------------ 训练数据集评估 ------------")
    # 模型评估
    y_train_predict = model.predict(X_train2)
    r2 = r2_score(y_train, y_train_predict)
    print("r2 score:", r2)
    mean_squared = mean_squared_error(y_train, y_train_predict)
    print("mean squared:", mean_squared)

    print("------------ 测试数据集评估 ------------")
    # 预测
    y_test_predcit = model.predict(X_test2)
    r2 = r2_score(y_test, y_test_predcit)
    print("r2 score:", r2)
    mean_squared = mean_squared_error(y_test, y_test_predcit)
    print("mean squared:", mean_squared)

    # 可视化
    plt.figure(figsize=(16, 9))
    plt.title("Polynomial Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, y, color="blue", marker="*", label="origin")
    plt.scatter(X_train, y_train_predict, marker="+", color="green", label="predict")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test01()
