#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

"""
X_train, X_test, y_train, y_test = 
sklearn.model_selection.train_test_split(
    *arrays, 
    test_size=None, 
    train_size=None, 
    random_state=None, 
    shuffle=True, 
    stratify=None)
python机器学习中常用 train_test_split()函数划分训练集和测试集
Parameters:
*arrays         sequence of indexables with same length / shape[0]
                Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.

test_size       float or int, default=None
                If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                If int, represents the absolute number of test samples. 
                If None, the value is set to the complement of the train size. 
                If train_size is also None, it will be set to 0.25.

train_size      float or int, default=None
                If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. 
                If int, represents the absolute number of train samples. 
                If None, the value is automatically set to the complement of the test size.

random_state    int, RandomState instance or None, default=None
                Controls the shuffling applied to the data before applying the split. 
                Pass an int for reproducible output across multiple function calls.

shuffle         bool, default=True
                Whether or not to shuffle the data before splitting. 
                If shuffle=False then stratify must be None.

stratify        array-like, default=None
                If not None, data is split in a stratified fashion, using this as the class labels.

Returns:        splitting list, length=2 * len(arrays)
                List containing train-test split of inputs.
                New in version 0.16: If the input is sparse, the output will be a scipy.sparse.csr_matrix. 
                Else, output type is the same as the input type.
"""


def test01():
    train_data = np.arange(0, 20)
    train_target = np.arange(20, 40)
    X_train, X_test, y_train, y_test = train_test_split(
        train_data, train_target, test_size=0.25, random_state=30
    )

    print(X_train)
    print("----------")
    print(X_test)
    print("----------")
    print(y_train)
    print("----------")
    print(y_test)


def test02():
    train_data = np.random.randint(0, 10, size=(20, 1))
    train_target = np.random.randint(0, 2, size=(20, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        train_data,
        train_target,
        test_size=0.25,
        random_state=train_data.shape[0],
        stratify=train_target,
    )
    print(X_train)
    print("----------")
    print(X_test)
    print("----------")
    print(y_train)
    print("----------")
    print(y_test)


if __name__ == "__main__":
    # test01()
    test02()
