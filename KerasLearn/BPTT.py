# -*- coding: UTF-8 -*-
#St = sigmod(U*x + W * s(t-1))
#Ot - sigmod(V*St)
#author:huyanglian
import numpy as np
from numpy import *
binary_dim = 8
hiden_dim = 100
input_dim = 2
output_dim = 1
W = np.random.rand(hiden_dim,hiden_dim)  #100个隐藏神经元
U = np.random.rand(input_dim,hiden_dim) #输入数据的维度
V = np.random.rand(hiden_dim,output_dim)


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

def Int2Binary(aint):
    abin = bin(aint)
    abinArray = zeros(8)
    index = 0
    for char in abin:
        if index > 1:
            abinArray[index-2] = int(char)
        index += 1
    return abinArray

#构造输入和输出序列

#St = sigmod(U*x + W * s(t-1))
for t in range(10000):
    aint = np.random.randint(0,127)
    abin = Int2Binary(aint)
    bint = np.random.randint(0, 127)
    bbin = Int2Binary(bint)

    cint = aint + bint
    cbin = Int2Binary(cint)
    x = np.array([abin,bbin]).T

    layer_1_values = list()
    layer_1_values.append(np.zeros(hiden_dim))

    for position in range(binary_dim):
        layer_1_value = sigmoid(np.dot(x[binary_dim - position - 1],U)+np.dot(layer_1_values[-1],W))
        y_predict = sigmoid(np.dot(layer_1_value,V))


    print(abin,bbin)

