#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pickle
import random


def serialize(obj, path):
    binary_data = pickle.dumps(obj)
    f = open(path, "wb")
    f.write(binary_data)
    f.close()


def unserialize(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def parse_data():
    labels = [
        "手机号码",
        "密码",
        "wxid",
        "a16",
        "安卓ID",
        "厂家",
        "型号",
        "安卓版本",
        "sdk版本",
        "hardware",
        "注册日期",
    ]
    df_csv_success = pd.read_csv("./data/data_success.csv", names=labels)
    df_csv_success["active"] = 1
    print("df_csv_success.shape:", df_csv_success.shape)
    # print(df_csv_success.head(10))

    df_csv_fail = pd.read_csv("./data/data_fail.csv", names=labels)
    df_csv_fail["active"] = 0
    print("df_csv_fail.shape:", df_csv_fail.shape)
    # print(df_csv_fail.head(10))

    df = pd.concat([df_csv_success, df_csv_fail], axis=0, ignore_index=True)
    print("df.shape:", df.shape)
    print(df.head(10))
    print(df.tail(10))

    # df_need = df.loc[:, ["active", "厂家", "型号", "安卓版本", "sdk版本", "hardware"]]
    # df_need.to_csv("./data/dataset01.csv", index=False)
    df.to_csv("./data/dataset.csv", index=False)


def train():
    df = pd.read_csv("./data/dataset.csv")
    # df["active"] = 0
    # print(df.head(10))
    # print(df.tail(10))

    print("------------- data -------------")
    df_X = df[["厂家", "型号", "安卓版本", "sdk版本", "hardware"]]
    df_y = df["active"]
    # df_y = np.zeros(df.shape[0]).reshape(-1, 1)
    # df_y = pd.DataFrame(df_y)
    print(df_X.shape)
    print(df_y.shape)
    # print(df_y2.shape)

    print("------------- OneHotEncoder -------------")
    # 创建编码器
    oe = OneHotEncoder()

    # OneHotEncoder要求输入为二维数组
    oe.fit(df_X)

    # 查看类别
    # print("classes: ", oe.categories_)
    # serialize(oe, "oe.pickle")

    # 调用transform获得编码结果
    encoded_labels = oe.transform(df_X).toarray()
    print(encoded_labels.shape)
    # print(encoded_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_labels, df_y, test_size=0.01, random_state=66, shuffle=True
    )
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # print(y_test)

    print("----------- train -----------")
    # 训练带有RBF核的SVM
    model = svm.SVC(kernel="rbf", gamma="auto").fit(X_train, y_train)
    # model = svm.SVC(kernel="rbf", gamma="auto").fit(df_X, df_y)
    # serialize(model, "model.pickle")
    # 获取回归截距
    print("intercept_:", model.intercept_)
    y_predict = model.predict(X_test)
    print("y_predict:\n", y_predict)
    print("y_test:\n", y_test)
    R2_score = r2_score(y_test, y_predict)
    print("r2 score:", R2_score)
    mean_squared = mean_squared_error(y_test, y_predict)
    print("mean_squared:", mean_squared)

    # print("----------- predict -----------")
    X_predict2 = df_X.loc[[random.randint(0, df.shape[0])]]
    print(X_predict2)
    encoded_labels2 = oe.transform(X_predict2).toarray()
    print(encoded_labels2.shape)
    y_predict2 = model.predict(encoded_labels2)
    print(y_predict2)


def test():
    data = [
        [
            "+66969290982",
            "fwso3689",
            "wxid_x4yg7yur8kri22",
            "A9c8c6d77ad6d668",
            "fa51097cb73d21d2",
            "HONOR",
            "DIO-AN00",
            12,
            32,
            "qcom",
            "2023-10-01 03:38:35",
        ]
    ]
    labels = [
        "手机号码",
        "密码",
        "wxid",
        "a16",
        "安卓ID",
        "厂家",
        "型号",
        "安卓版本",
        "sdk版本",
        "hardware",
        "注册日期",
    ]
    df = pd.DataFrame(data, columns=labels)
    # print(df)

    df_X = df[["厂家", "型号", "安卓版本", "sdk版本", "hardware"]]
    # print(df_X)

    oe = unserialize("oe.pickle")
    # print("classes: ", oe.categories_)

    # 调用transform获得编码结果
    encoded_labels = oe.transform(df_X.loc[[0]]).toarray()
    print(encoded_labels.shape)
    print(encoded_labels)

    model = unserialize("model.pickle")
    y_predict = model.predict(encoded_labels)
    print(y_predict)


def test01():
    labels = [
        "手机号码",
        "密码",
        "wxid",
        "a16",
        "安卓ID",
        "厂家",
        "型号",
        "安卓版本",
        "sdk版本",
        "hardware",
        "注册日期",
    ]
    df = pd.read_csv("./data/data_fail.csv", names=labels)

    df_X = df[["厂家", "型号", "安卓版本", "sdk版本", "hardware"]]

    oe = unserialize("oe.pickle")
    # print("classes: ", oe.categories_)

    # 调用transform获得编码结果
    encoded_labels = oe.transform(df_X.loc[[0]]).toarray()
    print(encoded_labels.shape)
    print(encoded_labels)

    model = unserialize("model.pickle")
    y_predict = model.predict(encoded_labels)
    print(y_predict)


if __name__ == "__main__":
    # parse_data()
    train()
    # test()
    # test01()
