#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

"""
线性回归
拟合线性回归模型
初始化模型 LinearRegression()
拟合 fit(x,y)
预测 predict(x_test)
"""


# 单因子线性回归预测
def test01():
    x = np.random.randint(0, 100, size=(1000, 1))
    y = 2 * x[:, 0] + 5

    # 创建线性回归模型
    model = LinearRegression()
    # 拟合数据
    model.fit(x, y)
    # 获取回归系数
    slope = model.coef_
    print("coef_:", slope)
    # 获取回归截距
    intercept = model.intercept_
    print("intercept_:", intercept)

    # 模型评佑
    y_new = model.predict(x)
    r2_score_error = r2_score(y, y_new)
    print("r2_score:", r2_score_error)
    squared_error = mean_squared_error(y, y_new)
    print("squared_error:", squared_error)

    # 预测
    x_test = np.array([[5]])
    y_test = model.predict(x_test)
    print(f"x_test:{x_test},y_test:{y_test}")

    # 可视化
    plt.figure(figsize=(16, 9))
    plt.title("Linear Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x[:, 0], y, color="blue", label="orign")
    plt.plot(x[:, 0], y_new, color="red", label="predict")
    plt.legend()
    plt.show()


# 多因子线性回归预测
def test02():
    x = np.random.randint(0, 100, size=(1000, 4))
    y = 3 * x[:, 0] + 10

    # 创建线性回归模型
    model = LinearRegression()
    # 拟合数据
    model.fit(x, y)
    # 获取回归系数
    slope = model.coef_
    print("coef_:", slope)
    # 获取回归截距
    intercept = model.intercept_
    print("intercept_:", intercept)

    # 模型评估
    y_new = model.predict(x)
    r2_score_error = r2_score(y, y_new)
    print("r2_score:", r2_score_error)
    squared_error = mean_squared_error(y, y_new)
    print("squared_error:", squared_error)

    # 预测
    x_test = np.array([[30, 34, 56, 65]])
    y_test = model.predict(x_test)
    print("x_test:", x_test)
    print("y_test:", y_test)

    # 绘图
    plt.figure(figsize=(16, 9))

    plt.subplot(221)
    plt.scatter(x[:, 0], y, color="red", linestyle="-", label="test1")
    plt.title("plot1")
    plt.legend()

    plt.subplot(222)
    plt.scatter(x[:, 1], y, color="green", linestyle="--", label="test2")
    plt.title("plot2")
    plt.legend()

    plt.subplot(223)
    plt.scatter(x[:, 2], y, color="blue", linestyle=":", label="test3")
    plt.title("plot3")
    plt.legend()

    plt.subplot(224)
    plt.scatter(x[:, 3], y, color="yellow", linestyle="-.", label="test4")
    plt.title("plot4")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test01()
    # test02()
