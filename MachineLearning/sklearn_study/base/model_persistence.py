#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sklearn.externals as sk_externals
from sklearn.linear_model import LinearRegression

model = LinearRegression()
sk_externals.joblib.dump(model, "model.pickle")  # 保存
model = sk_externals.joblib.load("model.pickle")  # 载入
