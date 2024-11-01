#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.decomposition import PCA

"""
PCA（Principal Component Analysis）是一种常用的数据分析方法。PCA通过线性变换将原始数据变换为一组各维度线性无关的表示，
可用于提取数据的主要特征分量，常用于高维数据的降维

基于SVD分解协方差矩阵实现PCA算法
输入：数据集 ，需要降到k维。

1) 去平均值，即每一位特征减去各自的平均值。

2) 计算协方差矩阵。

3) 通过SVD计算协方差矩阵的特征值与特征向量。

4) 对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵。

5) 将数据转换到k个特征向量构建的新空间中。

在PCA降维中，我们需要找到样本协方差矩阵 
 的最大k个特征向量，然后用这最大的k个特征向量组成的矩阵来做低维投影降维。可以看出，在这个过程中需要先求出协方差矩阵 
,当样本数多、样本特征数也多的时候，这个计算还是很大的。当我们用到SVD分解协方差矩阵的时候，SVD有两个好处：

1) 有一些SVD的实现算法可以先不求出协方差矩阵 
 也能求出我们的右奇异矩阵V。也就是说，我们的PCA算法可以不用做特征分解而是通过SVD来完成，这个方法在样本量很大的时候很有效。实际上，scikit-learn的PCA算法的背后真正的实现就是用的SVD，而不是特征值分解。

2)注意到PCA仅仅使用了我们SVD的左奇异矩阵，没有使用到右奇异值矩阵，那么右奇异值矩阵有什么用呢？

假设我们的样本是m*n的矩阵X，如果我们通过SVD找到了矩阵 
 最大的k个特征向量组成的k*n的矩阵 
 ,则我们可以做如下处理：


可以得到一个m*k的矩阵X',这个矩阵和我们原来m*n的矩阵X相比，列数从n减到了k，可见对列数进行了压缩。也就是说，左奇异矩阵可以用于对行数的压缩；右奇异矩阵可以用于对列(即特征维度)的压缩。这就是我们用SVD分解协方差矩阵实现PCA可以得到两个方向的PCA降维(即行和列两个方向)。
"""


def pca(X, k):  # k is the components you want
    # mean of each feature
    _, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data


def test01():
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X_pca = pca(X, 1)
    print(X_pca.shape)
    print(X_pca)


def test02():
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    model = PCA(n_components=1)
    model.fit(X)
    X_pca = model.transform(X)
    print(X_pca.shape)
    print(X_pca)


if __name__ == "__main__":
    test01()
    test02()
