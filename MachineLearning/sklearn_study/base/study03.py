#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from sklearn.preprocessing import LabelEncoder

"""
标签编码(label encoding): 将分类变量的类别编码为数字。

用sklearn.preprocessing.LabelEncoder实现，将包含k个类别的分类变量编码为$0,1,2,...(k-1)$

标签编码一般不用于特征，而是用于目标变量。

假设一个代表性别的特征'gender'，包含两个类别：'male','female'。标签编码将类别编码为整数，0代表男性，1代表女性
。但这不符合模型背后的假设，因为机器学习模型认为数据有算数含义，例如0 < 1，这意味着男性 < 女性，但这种关系不成立。
一般会使用独热编码处理分类特征，标签编码仅用于分类目标。
"""


def test01():
    # 目标变量
    y = ["up", "up", "down", "range", "up", "down", "range", "range", "down"]

    # 创建LabelEncoder对象
    le = LabelEncoder()

    # 拟合数据
    le.fit(y)

    # 查看包含哪些类别
    print("classes: ", le.classes_)

    # 编码为数字
    print("encoded labels: ", le.transform(y))

    # 调用inverse_transform实现反向操作
    print("inverse encoding: ", le.inverse_transform([0, 1, 2]))


if __name__ == "__main__":
    test01()
