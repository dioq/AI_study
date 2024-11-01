#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 创建一个同心圆的数据集
def create_concentric_circles(n_samples=100):
    np.random.seed(0)
    X_inner = np.random.randn(n_samples // 2, 2) * 0.3
    X_outer = np.random.randn(n_samples // 2, 2) * 0.5 + 1.5
    X = np.vstack([X_inner, X_outer])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return X, y


# 绘制数据点和决策边界
def plot_decision_boundary(classifier, X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # 创建网格来评估模型
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy).reshape(XX.shape)
    # 绘制决策边界和边界
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    ax.set_title(title)


def test01():
    # 生成数据
    X, y = create_concentric_circles(800)
    # 训练线性SVM
    linear_svm = svm.SVC(kernel="linear").fit(X, y)
    y_predict = linear_svm.predict(X)
    r2_score_error = r2_score(y, y_predict)
    print("r2_score:", r2_score_error)
    squared_error = mean_squared_error(y, y_predict)
    print("squared_error:", squared_error)
    # 训练带有RBF核的SVM
    model = svm.SVC(kernel="rbf", gamma="auto").fit(X, y)
    y_predict = model.predict(X)
    r2_score_error = r2_score(y, y_predict)
    print("r2_score:", r2_score_error)
    squared_error = mean_squared_error(y, y_predict)
    print("squared_error:", squared_error)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(linear_svm, X, y, "Linear SVM")
    plt.subplot(1, 2, 2)
    plot_decision_boundary(model, X, y, "RBF Kernel SVM")
    plt.show()


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


def test02():
    X, y = target_data(10000)

    # 训练线性SVM
    linear_svm = svm.SVC(kernel="linear").fit(X, y)
    # 获取回归系数
    print("coef_:", linear_svm.coef_)
    # 获取回归截距
    print("intercept_:", linear_svm.intercept_)
    y_predict = linear_svm.predict(X)
    r2_score_error = r2_score(y, y_predict)
    print("r2_score:", r2_score_error)
    squared_error = mean_squared_error(y, y_predict)
    print("squared_error:", squared_error)
    # 训练带有RBF核的SVM
    model = svm.SVC(kernel="rbf", gamma="auto").fit(X, y)
    # 获取回归系数
    # print("coef_:", model.coef_)
    # 获取回归截距
    print("intercept_:", model.intercept_)
    y_predict = model.predict(X)
    r2_score_error = r2_score(y, y_predict)
    print("r2_score:", r2_score_error)
    mean_squared = mean_squared_error(y, y_predict)
    print("mean_squared:", mean_squared)

    x_test = [[98, 103]]
    y_test = model.predict(x_test)
    print(f"x_test:{x_test},y_test:{y_test}")

    # 绘图
    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(linear_svm, X, y, "Linear SVM")
    plt.subplot(1, 2, 2)
    plot_decision_boundary(model, X, y, "RBF Kernel SVM")
    plt.show()


def test03():
    # 创建特征列表表头
    column_names = [
        "Sample code number",
        "Clump Thickness",
        "Uniformity of Cell Size",
        "Uniformity of Cell Shape",
        "Marginal Adhesion",
        "Single Epithelial Cell Size",
        "Bare Nuclei",
        "Bland Chromatin",
        "Normal Nucleoli",
        "Mitoses",
        "Class",
    ]
    # 使用pandas.read_csv函数从网上读取数据集
    data = pd.read_csv("./DATA/data.csv", names=column_names)
    # 将？替换为标准缺失值表示
    data = data.replace(to_replace="?", value=np.nan)
    # 丢弃带有缺失值的数据(只要有一个维度有缺失便丢弃)
    data = data.dropna(how="any")
    # 查看data的数据量和维度
    print("data.shape:", data.shape)

    # 随机采样25%的数据用于测试，剩下的75%用于构建训练集
    X_train, X_test, y_train, y_test = train_test_split(
        data[column_names[1:10]],
        data[column_names[10]],
        test_size=0.25,
        random_state=33,
    )
    # 查看训练样本的数量和类别分布
    print("y_train.value_counts():", y_train.value_counts())

    # 查看测试样本的数量和类别分布
    print("y_test.value_counts():", y_test.value_counts())

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # 训练带有RBF核的SVM
    model = svm.SVC(kernel="rbf", gamma="auto").fit(X_train, y_train)
    # 获取回归系数
    # print("coef_:", model.coef_)
    # 获取回归截距
    print("intercept_:", model.intercept_)
    y_predict = model.predict(X_test)
    R2_score = r2_score(y_test, y_predict)
    print("r2 score:", R2_score)
    mean_squared = mean_squared_error(y_test, y_predict)
    print("mean_squared:", mean_squared)


def test04():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    model = svm.SVC(
        C=1.0,
        cache_size=200,
        class_weight=None,
        coef0=0.0,
        decision_function_shape="ovr",
        degree=3,
        gamma="auto",
        kernel="rbf",
        max_iter=-1,
        probability=False,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False,
    )
    model.fit(X, y)
    # 获取回归截距
    print("intercept_:", model.intercept_)
    y_predict = model.predict(X)
    R2_score = r2_score(y, y_predict)
    print("r2 score:", R2_score)
    mean_squared = mean_squared_error(y, y_predict)
    print("mean_squared:", mean_squared)
    print(model.predict([[-0.8, -1]]))


if __name__ == "__main__":
    # test01()
    test02()
    # test03()
    # test04()
