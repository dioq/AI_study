#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler  # 标准化工具


def test01():
    x_np = np.array([[1.5, -1.0, 2.0], [2.0, 0.0, 0.0]])
    mean = np.mean(x_np, axis=0)
    std = np.std(x_np, axis=0)
    print("矩阵初值为:\n", x_np)
    print("该矩阵的均值为:\n{}\n该矩阵的标准差为:\n{}".format(mean, std))
    another_trans_data = x_np - mean
    another_trans_data = another_trans_data / std
    print("标准差标准化的矩阵为:\n{}".format(another_trans_data))


def test02():
    x_np = np.array([[1.5, -1.0, 2.0], [2.0, 0.0, 0.0]])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_np)
    print("矩阵初值为:\n{}".format(x_np))
    print(
        "该矩阵的均值为:\n{}\n 该矩阵的标准差为:\n{}".format(
            scaler.mean_, np.sqrt(scaler.var_)
        )
    )
    print("标准差标准化的矩阵为:\n{}".format(x_train))


if __name__ == "__main__":
    # test01()
    test02()
