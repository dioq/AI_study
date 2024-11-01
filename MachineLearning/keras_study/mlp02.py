#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist  # google 提供的手写数字图片集
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error, r2_score  # , accuracy_score
from matplotlib import pyplot as plt

"""
多分类问题,识别手写数字
"""


def test01():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(type(X_train), X_train.shape)

    img1 = X_train[0]
    print(img1.shape)  # 图片像素 28 * 28

    # fig1 = plt.figure(figsize=(5, 5))
    # plt.imshow(img1)
    # plt.title(y_train[0])
    # plt.show()

    # format the input data
    feature_size = img1.shape[0] * img1.shape[1]
    X_train_format = X_train.reshape(X_train.shape[0], feature_size)
    X_test_format = X_test.reshape(X_test.shape[0], feature_size)
    print(X_train_format.shape)
    print(X_test_format.shape)

    # normalize the input data
    X_train_normal = X_train_format / 255
    X_test_normal = X_test_format / 255
    # print(X_train_normal.shape)
    # print(X_test_normal.shape)

    # format the output data(labels)
    y_train_format = to_categorical(y_train)
    y_test_format = to_categorical(y_test)
    print(y_train.shape)
    print(y_train[0])
    print(y_train_format.shape)
    print(y_train_format[0])

    # set up model
    mlp = Sequential()
    mlp.add(Dense(units=392, activation="sigmoid", input_dim=feature_size))
    mlp.add(Dense(units=392, activation="sigmoid"))
    mlp.add(Dense(units=10, activation="softmax"))

    mlp.summary()

    # upload_dir the model
    mlp.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train the model
    mlp.fit(X_train_normal, y_train_format, epochs=3)

    # evaluate the model
    # 训练数据集评估
    y_train_predict = mlp.predict(X_train_normal)
    r2 = r2_score(y_train_format, y_train_predict)
    print("r2_score:", r2)
    # accuracy = accuracy_score(y_train_format, y_train_predict)
    # print("accuracy:", accuracy)
    mean_squared = mean_squared_error(y_train_format, y_train_predict)
    print("mean_squared:", mean_squared)

    # 测试数据集评估
    y_test_predict = mlp.predict(X_test_normal)
    r2 = r2_score(y_test_format, y_test_predict)
    print("r2_score:", r2)
    mean_squared = mean_squared_error(y_test_format, y_test_predict)
    print("mean_squared:", mean_squared)

    # one for test
    num_test = 100
    X_num_test = X_test[num_test]
    X_num_test_format = X_num_test.reshape(1, feature_size)
    X_num_test_normal = X_num_test_format / 255
    label_test_predict = mlp.predict(X_num_test_normal)

    for i in range(0, label_test_predict.shape[1]):
        print(i, ":", label_test_predict[0][i])

    fig2 = plt.figure(figsize=(5, 5))
    plt.imshow(X_num_test)
    # plt.title()
    plt.show()


if __name__ == "__main__":
    test01()
