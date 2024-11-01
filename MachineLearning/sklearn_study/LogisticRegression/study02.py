#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

# 使用sklearn.cross_validation里的train_test_split模块分割数据集
from sklearn.model_selection import train_test_split

# 从sklearn.preprocessing导入StandardScaler
from sklearn.preprocessing import StandardScaler

# 从sklearn.linear_model导入LogisticRegression（逻辑斯蒂回归）
from sklearn.linear_model import LogisticRegression

# 从sklearn.linear_model导入SGDClassifier（随机梯度参数）
from sklearn.linear_model import SGDClassifier

# 从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score


def test01():
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
    data = pd.read_csv("../DATA/data.csv", names=column_names)
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

    lr = LogisticRegression()
    # 调用逻辑斯蒂回归，使用fit函数训练模型参数
    lr.fit(X_train, y_train)
    # 获取回归系数
    coef = lr.coef_
    print("coef_:", coef)
    # 获取回归截距
    intercept = lr.intercept_
    print("intercept_:", intercept)

    y_predict = lr.predict(X_test)
    # 调用随机梯度的fit函数训练模型
    # print("y_predict:", y_predict)
    R2_score = r2_score(y_test, y_predict)
    print("r2 score:", R2_score)
    mean_squared = mean_squared_error(y_test, y_predict)
    print("mean_squared:", mean_squared)

    # 使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果
    print("Accuracy of LR Classifier:", lr.score(X_test, y_test))
    # 使用classification_report模块获得逻辑斯蒂模型其他三个指标的结果（召回率，精确率，调和平均数）
    print(
        classification_report(y_test, y_predict, target_names=["Benign", "Malignant"])
    )


if __name__ == "__main__":
    test01()
