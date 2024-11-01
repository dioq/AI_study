#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch as t


def test01():
    x = t.rand(5, 3)
    print(x)
    y = t.rand(5, 3)
    print(y)
    # 在不支持CUDA的机器下,下一步不会运行
    if t.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    print(x + y)


if __name__ == "__main__":
    test01()
