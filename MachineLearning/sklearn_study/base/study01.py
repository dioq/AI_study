#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
函数原型
sklearn.datasets.load_iris(*, return_X_y=False, as_frame=False)
函数说明
Iris数据集是常用的分类实验数据集,由Fisher, 1936收集整理。Iris也称鸢尾花卉数据集,是一类多重变量分析的数据集。
数据集包含150个数据样本,分为3类,每类50个数据,每个数据包含4个属性(分别是：花萼长度,花萼宽度,花瓣长度,花瓣宽度)。
可通过这4个属性预测鸢尾花卉属于(Setosa,Versicolour,Virginica)三个种类的鸢尾花中的哪一类。
Iris里有两个属性iris.data,iris.target。data是一个矩阵,每一列代表了萼片或花瓣的长宽,一共4列,
每一列代表某个被测量的鸢尾植物,一共有150条记录。

参数
return_X_y  为True, 则返回一个(data, target)元组类型的数据
            为False,则返回一个类似字典的Bunch对象
as_frame    为True, 则表示返回的data、target类型为DataFrame类型
            为False,则返回的类型为ndarray数组类型
"""


def test01():
    dataset = datasets.load_iris(return_X_y=True)
    print(dataset)


def test02():
    dataset = datasets.load_iris(return_X_y=True, as_frame=True)
    print(dataset)


def test03():
    dataset = datasets.load_iris(return_X_y=False, as_frame=False)
    _, ax = plt.subplots()
    scatter = ax.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target)
    ax.set(xlabel=dataset.feature_names[0], ylabel=dataset.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0],
        dataset.target_names,
        loc="lower right",
        title="Classes",
    )
    plt.show()


def test04():
    iris = datasets.load_iris()

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=iris.target,
        s=40,
    )

    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])

    plt.show()


if __name__ == "__main__":
    # test01()
    # test02()
    test03()
    # test04()
