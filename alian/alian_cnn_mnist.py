# -*- coding: UTF-8 -*-
#卷积神经网络 python实现 ，对mnist分类
# softmax偏导数求解过程：http://www.jianshu.com/p/ffa51250ba2e
#C++ 实现的版本 http://blog.csdn.net/shangming111/article/details/41082631
#https://www.cnblogs.com/alexanderkun/p/4863691.html


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from scipy.signal import convolve2d

import time
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
#sudo pip install h5py

a = np.array([[0,0,0,0],[0,1,3,0],[0,2,2,0],[0,0,0,0]])
b= np.array([[0.1,0.2],[0.2,0.4]])
print(convolve2d(a,b,mode='valid'))
#import h5py
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

X_train = X_train[0:2000]
y_train = y_train[0:2000]
EPOCH_NUM = 1
#定义卷积核以及其他相关参数

#32个5*5的卷积核
C1 = np.random.rand(32,5,5)
C1_Out = np.zeros((32,28,28))

#池化层输出
S2_out = np.zeros((32,14,14))

#64个5*5的卷积核
C3 = np.random.rand(64,5,5)
C3_Out = np.zeros((64,14,14))

#池化层输出
S4_out = np.zeros((64,7,7))

# 全连接1：将S3_out进行光栅化变成(64 * 7 * 7) = (3136)，并且将向量长度降到1024
Flat1 = np.random.rand(3136,)
W1 = np.random.rand(3136,1024)


#全连接2:将长度为1024的向量降维到10
Flat2 = np.random.rand(1024,)
W2 = np.random.rand(1024,10)

Vector10 = np.random.rand(10,)

def sigmoid(x):
    #print("sigmod x=",x)
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

print(sigmoid(0))

a = np.array([[0,0,0,0],[0,1,3,0],[0,2,2,0],[0,0,0,0]])
b= np.array([[0.1,0.2],[0.2,0.4]])
print(convolve2d(a,b,mode='valid'))
#对图像进行卷积，边缘填充为0
def _convolve(img, fil,img_out):
    fil_heigh = fil.shape[0]  # 获取卷积核(滤波)的高度
    fil_width = fil.shape[1]  # 获取卷积核(滤波)的宽度

    conv_heigh = img.shape[0]
    conv_width = img.shape[1]
    #conv = np.zeros((conv_heigh, conv_width))
    starttime = datetime.datetime.now()
    for i in range(conv_heigh):
        for j in range(conv_width):  # 逐点相乘并求和得到每一个点
            temp = 0
            for fil_i in range(fil_heigh):
                for fil_j in range(fil_width):
                    temp_h = i - int(conv_heigh/2) + fil_i
                    temp_w = j - int(conv_width/2) + fil_j
                    if temp_h >=0 and temp_w >= 0:
                        temp += img[temp_h][temp_w] * fil[fil_i][fil_j]

            img_out[i][j] = temp
    endtime = datetime.datetime.now()
    #print("convcostttime==",(endtime-starttime).microseconds)
#下采样，用maxpooling
def maxpolling(img, subsize,img_out):
    conv_heigh = img_out.shape[0]
    conv_width = img_out.shape[1]
    #conv = np.zeros((conv_heigh, conv_width))
    for i in range(conv_heigh):
        for j in range(conv_width):  # 逐点相乘并求和得到每一个点
            max = -1000000
            for nSubSizeIndex_x  in range(subsize):
                for nSubSizeIndex_y in range(subsize):
                    if(img[i *subsize + nSubSizeIndex_x][j *subsize + nSubSizeIndex_y] > max):
                        img_out[i][j] = img[i *subsize + nSubSizeIndex_x][j *subsize + nSubSizeIndex_y]
                        max = img_out[i][j]


for nEpoch in range(EPOCH_NUM):
    for nTrainIndex in range(X_train.shape[0]):
        Img = X_train[nTrainIndex][0]
        #第一层卷积
        #C1_Out_Sum = np.zeros((32,28,28))
        for i in range(32):
            a = convolve2d(Img,C1[i])
            C1_Out[i] = convolve2d(Img,C1[i],boundary='symm',mode='same')
            C1_Out[i] = sigmoid(C1_Out[i])
            #C1_Out_Sum += C1_Out[i]

        #下采样
        for i in range(32):
            maxpolling(C1_Out[i],2,S2_out[i])

        #tempC3_Sum = np.zeros((14, 14))
        tempC3 = np.zeros((14, 14))

        #再做第二层卷积

        starttime = datetime.datetime.now()
        # long running
        for C3_index in range(64):
            tempC3_Sum = np.zeros((14,14))
            for S2_index in range(32):
                #tempC3 = np.zeros((14,14))
                tempC3 = convolve2d(S2_out[S2_index],C3[C3_index],boundary='symm',mode='same')
                tempC3_Sum += tempC3
            C3_Out[C3_index] = sigmoid(tempC3_Sum)
        endtime = datetime.datetime.now()
        print((endtime-starttime).microseconds)
            # 下采样
        for i in range(64):
            maxpolling(C3_Out[i], 2, S4_out[i])

        #把S4光栅化
        for i in range(64):
            for h in range(7):
                for w in range(7):
                    Flat1[i* 49 + h * 7 + w] = S4_out[i][h][w]

        Flat2 = Flat1.dot(W1)
        Vector10 = Flat2.dot(W2)
        print("alian")






def wise_element_sum(img,fil):
    for i in range(32):
        C1_Out[i] = _convolve()






#全连接层输出





#把训练值转为one-hot向量
def toOneHot(nLable,nClasses):
    one_hot = np.zeros(nClasses)
    for index in(nClasses):
        if(nLable == index):
            one_hot[index] = 1
        else:
            one_hot[index] = 0
    return one_hot

def ff():

    return 0