#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch

"""
tensor.detach()
返回一个新的tensor,从当前计算图中分离下来的,但是仍指向原变量的存放位置,不同之处只是requires_grad为false,
得到的这个tensor永远不需要计算其梯度,不具有grad。
即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
这样我们就会继续使用这个新的tensor进行计算,后面当我们进行反向传播时,到该调用detach()的tensor就会停止,不能再继续向前进行传播
注意：
使用detach返回的tensor和原始的tensor共同一个内存,即一个修改另一个也会跟着改变。
"""


a = torch.tensor([1, 2, 3.0], requires_grad=True)
print(a.grad)
out = a.sigmoid()
print(out)

# 添加detach(),c的requires_grad为False
c = out.detach()
print(c)

# 这时候没有对c进行更改,所以并不会影响backward()
out.sum().backward()
print(a.grad)
