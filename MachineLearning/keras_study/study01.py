#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from keras.utils import to_categorical

"""
to_categorical()
将类别向量转换为二进制（只有0和1）的矩阵类型表示。
其表现为将原有的类别向量转换为独热编码的形式。
"""


def test01():
    # 类别向量定义
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    print(len(labels))
    hotcode = to_categorical(labels)
    print(hotcode)


if __name__ == "__main__":
    test01()
