# -*- coding: UTF-8 -*-
#卷积神经网络 python实现 ，对mnist分类
# softmax偏导数求解过程：http://www.jianshu.com/p/ffa51250ba2e
#C++ 实现的版本 http://blog.csdn.net/shangming111/article/details/41082631

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
#sudo pip install h5py
import h5py
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

EPOCH_NUM = 1
#定义卷积核以及其他相关参数

#32个5*5的卷积核
C1 = np.random.rand(32,5,5)
C1_Out = np.zeros(32,28,28)

#池化层输出
S2_out = np.zeros(32,14,14)

#64个5*5的卷积核
C3 = np.random.rand(64,5,5)
C3_Out = np.zeros(64,14,14)

#池化层输出
S3_out = np.zeros(64,7,7)

# 全连接1：将S3_out进行光栅化变成(64 * 7 * 7) = (3136)，并且将向量长度降到1024
Flat1 = np.random.rand(1,3136)
W1 = np.random.rand(3136,1024)

#全连接2:将长度为1024的向量降维到10
W2 = np.random.rand(1024,10)

#对图像进行卷积，边缘填充为0
def _convolve(img, fil):
    fil_heigh = fil.shape[0]  # 获取卷积核(滤波)的高度
    fil_width = fil.shape[1]  # 获取卷积核(滤波)的宽度

    conv_heigh = img.shape[0]
    conv_width = img.shape[1]
    conv = np.zeros((conv_heigh, conv_width))

    for i in range(conv_heigh):
        for j in range(conv_width):  # 逐点相乘并求和得到每一个点
            temp = 0
            for fil_i in range(fil_heigh):
                for fil_j in range(fil_width):
                    temp_h = i - int(conv_heigh/2)  + fil_i
                    temp_w = j - int(conv_width/2)  + fil_j
                    if temp_h >=0 and temp_w >= 0:
                        temp += img[temp_h][temp_w] * fil[fil_i][fil_j]

            conv[i][j] = temp
    return conv

def wise_element_sum(img,fil):
    return (img * fil).sum()

for i in range(EPOCH_NUM):
    # 正向传播,计算出10维的向量




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