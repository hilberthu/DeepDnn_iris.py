# -*- coding: UTF-8 -*-
#St = sigmod(U*x + W * s(t-1))
#Ot - sigmod(V*St)
#author:huyanglian
#实现了两层神经网络，模拟y=(x1+x2)/2的运算
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
EPOCHNUM = 30
BATCHNUM = 100
TAINING_LEN = 10000
neroNums = 35
np.random.seed(0)

W1 = np.random.rand(2,neroNums) #
W2 = np.random.rand(neroNums,)  #1
X=zeros((TAINING_LEN,2))

B1 = zeros(neroNums,)
B2 = 0

Y = zeros((TAINING_LEN,1))
LEARN_RATE = 0.01
temp = (np.dot(np.array(X[0]),W1))
def sigmoid(x):
    #print("sigmod x=",x)
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

'''def sigmoid(x):
    return np.tanh(x)


def sigmoid_output_to_derivative(x):
    return 1.0 - np.tanh(x) * np.tanh(x)
'''

for j in range(TAINING_LEN):
    X[j][0] = float(np.random.randint(100 ))/100  # int version
    X[j][1] = float(np.random.randint(100 ))/100  # int version
    Y[j] = (X[j][0] + X[j][1]) / 2

for nEpoch in range(EPOCHNUM):
    nBatchNum = 0
    nTotalEM = 0
    for nBatchIndex in range(TAINING_LEN/BATCHNUM):
        #正向传播,计算误差
        deltaW2 = zeros((neroNums,))
        deltaW1 = zeros((2,neroNums))
        deltaB2 = 0
        for nInnerIndex in range(BATCHNUM):
            x = np.array(X[nBatchIndex * BATCHNUM + nInnerIndex])
            y = Y[nBatchIndex * BATCHNUM + nInnerIndex]
            temp1 = np.dot(x,W1)
            l1 = sigmoid(temp1)
            y_predict = sigmoid(np.dot(l1.T,W2))
            #[ 0.67628315  0.61915201  0.61756472][ 0.97919739  0.13973793  0.08555134]
            nTotalEM = nTotalEM + pow((y_predict - y[0]),2)
            deltaTemp = (y_predict -y[0])*sigmoid_output_to_derivative(y_predict)
            deltaW2 +=  deltaTemp*l1
            #deltaB2 += (y_predict -y[0])*sigmoid_output_to_derivative((y_predict))

            for i in range(2):
                for j in range(neroNums):
                    deltaW1[i][j] += deltaTemp*W2[j]*sigmoid_output_to_derivative(l1[j])*x[i]

        W2 = W2 - LEARN_RATE * deltaW2
        W1 = W1 - LEARN_RATE * deltaW1
    print("epoch", nEpoch, nTotalEM / TAINING_LEN)
        #B2+=deltaB2
#test
Y_Predict = np.zeros(100)
Y_REAL = np.zeros(100)
for j in range(100):
    x1 = float(np.random.randint(100)) / 100  # int version
    x2 = float(np.random.randint(100)) / 100  # int version
    y = (x1 + x2) / 2
    l1 = sigmoid(np.dot(np.array([x1,x2]), W1))
    y_predict = sigmoid(np.dot(l1.T, W2))
    print("y_predict==", y_predict, y)
    Y_Predict[j] = y_predict
    Y_REAL[j] = y

plt.plot(Y_Predict,'b',  label='Predict')
plt.plot(Y_REAL, 'r',label='Real')
plt.show()