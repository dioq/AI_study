#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""
独热编码(one hot encoding): 将包含m个类别的分类变量转化为$n*m$的二元矩阵，n是观测值数量，m是类别数量。
假设分类变量'car_type'，表示汽车类型，包含类别(BMW, Tesla, Audi)，独热编码将产生下图的结果，
每一个类别都成为一个新的变量/特征，1表示观测值包含该类别，0表示不包含。
"""


def test01():
    # 分类特征
    data = np.array(["BMW", "Tesla", "Audi", "BMW", "Audi"])

    # 创建编码器
    oe = OneHotEncoder()

    # OneHotEncoder要求输入为二维数组，用reshape重排结构
    oe.fit(data.reshape(-1, 1))

    # 查看类别
    print("classes: ", oe.categories_)

    # 调用transform获得编码结果
    # transform默认返回稀疏矩阵，当分类变量包含很多类别时非常有用，进一步调用toarray可获得熟悉的numpy二维数组
    encoded_labels = oe.transform(data.reshape(-1, 1)).toarray()
    print(encoded_labels)

    # 用数据框展现最终的结果，便于理解
    encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_)
    print(encoded_labels_df)


def test02():
    """
    调用pd.get_dummies实现独热编码，这比sklearn的接口更方便，因为它允许我们直接操作数据框。
    pd.get_dummies默认将数据类型为'object'的变量视为分类变量，也可以提供要编码的变量名称。
    """
    data = pd.DataFrame({"car_type": ["BMW", "Tesla", "Audi", "BMW", "Audi"]})
    print(data)

    data_encoded = pd.get_dummies(data)
    print(data_encoded)


def test03():
    # 分类特征
    data = np.array(
        [
            [165349.20, 136897.80, 471784.10, "纽约", 192261.83],
            [162597.70, 151377.59, 443898.53, "加州", 191792.06],
            [153441.51, 101145.55, 407934.54, "佛罗里达州", 191050.39],
            [144372.41, 118671.85, 383199.62, "纽约", 182901.99],
            [142107.34, 91391.77, 366168.42, "佛罗里达州", 166187.94],
            [131876.90, 99814.71, 362861.36, "纽约", 156991.12],
            [134615.46, 147198.87, 127716.82, "加州", 156122.51],
            [130298.13, 145530.06, 323876.68, "佛罗里达州", 155752.60],
            [120542.52, 148718.95, 311613.29, "纽约", 152211.77],
            [123334.88, 108679.17, 304981.62, "加州", 149759.96],
        ]
    )
    df = pd.DataFrame(
        data, columns=["研发支出", "行政管理", "营销支出", "State", "盈利"]
    )
    print("df:\n", df)

    # creating instance of one-hot-encoder
    oe = OneHotEncoder()
    state_column = df[["State"]]
    print("state_column:\n", state_column)
    oe.fit(state_column)
    print("classes: ", oe.categories_)

    # 查看类别
    encoded_labels = oe.transform(state_column)
    print(encoded_labels)
    print(encoded_labels.shape)
    print(encoded_labels.toarray())

    enc_df = pd.DataFrame(encoded_labels.toarray(), columns=oe.categories_[0])
    print("enc_df:\n", enc_df)
    # merge with main df bridge_df on key values
    df.drop("State", axis=1, inplace=True)
    df = df.join(enc_df)
    print("df:\n", df)

    columns1 = ["研发支出", "行政管理", "盈利"]
    columns_need = columns1 + list(oe.categories_[0])
    print(columns_need)
    X = df.loc[:, columns_need]
    y = df.loc[:, ["营销支出"]]
    print("X:\n", X)
    print("y:\n", y)

    # 处理待预测数据
    X_predict = [["纽约"]]
    encoded_labels = oe.transform(X_predict)
    print(X_predict)
    # print(encoded_labels)
    print(encoded_labels.shape)
    print(encoded_labels.toarray())


def test04():
    # 分类特征
    data = np.array(
        [
            ["AA1", "BB1", "CC1", "DD1", "EE1"],
            ["AA2", "BB2", "CC2", "DD2", "EE2"],
            ["AA1", "BB1", "CC1", "DD1", "EE1"],
        ]
    )
    print(data)
    # 创建编码器
    oe = OneHotEncoder()

    # OneHotEncoder要求输入为二维数组
    oe.fit(data)

    # 查看类别
    print("type:", type(oe.categories_))
    print("classes: ", oe.categories_)

    # 调用transform获得编码结果
    # transform默认返回稀疏矩阵，当分类变量包含很多类别时非常有用，进一步调用toarray可获得熟悉的numpy二维数组
    # column = np.array(data)
    encoded_labels = oe.transform(data).toarray()
    print(encoded_labels)
    print(encoded_labels.shape)

    # 用数据框展现最终的结果，便于理解
    column_list = []
    for item in oe.categories_:
        print(item)
        # column_list = column_list + item
        column_list.append(item[0])
        column_list.append(item[1])
    encoded_labels_df = pd.DataFrame(encoded_labels, columns=column_list)
    print(encoded_labels_df)

    X_predict = [["AA2", "BB1", "CC2", "DD1", "EE2"]]
    encoded_labels = oe.transform(X_predict).toarray()
    print("----------- predict -----------")
    print(encoded_labels.shape)
    print(encoded_labels)


if __name__ == "__main__":
    # test01()
    # test02()
    # test03()
    test04()
    # test05()
