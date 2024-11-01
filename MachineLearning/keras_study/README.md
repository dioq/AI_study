# 机器学习

## Multi-layer perceptron  多层感知机 mlp

Dense(units=20, input_dim=2, activation="sigmoid")
参数:
units                           神经元数量
input_dim                       输入参数维度
activation                      激活函数

model = Sequential()
model.compile(optimizer="adam", loss="binary_crossentropy")
参数:
optimizer                       优化器
loss                            损失函数
                                binary_crossentropy             二分类损失函数
                                categorical_crossentropy        多分类损失函数
