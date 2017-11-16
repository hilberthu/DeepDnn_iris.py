# -*- coding: UTF-8 -*-
"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 6 - CNN example

# to try tensorflow, un-comment following two lines
# import os
# os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from PIL import Image
import os
import matplotlib.pyplot as plt
from keras.models import load_model

X_train  = np.zeros((500,1,64,64))
Y_train  = np.zeros((500,15))
list_paths = []
list_imagefilename = []
nTotalImages = 0
def loadImages():
    rootdir_image = "/home/alian/PycharmProjects/DeepDnn_iris.py/alian/yalefaces"
    global list_paths
    list_paths = os.listdir(rootdir_image)
    for i in range(len(list_paths)):
        path = os.path.join(rootdir_image, list_paths[i])
        if path.find("leftlight") < 0:
            continue

        im = Image.open(path)
        im = im.resize((64,64))
        pix = im.load()
        width = im.size[0]
        height = im.size[1]
        image_pix = np.zeros((height,width))
        for x in range(width):
            for y in range(height):
                temp = pix[x,y]
                image_pix[y][x] = temp
        y = list_paths[i][7:9]
        Y_train[nTotalImages] = np_utils.to_categorical(int(y)-1, num_classes=15)
        X_train[nTotalImages][0] = image_pix/255
        global list_imagefilename
        list_imagefilename.append(list_paths[i])
        global nTotalImages
        nTotalImages = nTotalImages + 1


        #plt.imshow(image_pix,cmap ='gray')
        #plt.show()

loadImages()
#sudo pip install h5py
import h5py
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X_train = X_train[:10000]
#y_train = y_train[:10000]
# Another way to build your CNN
model = load_model("yale_model.h5")
predict_result = model.predict(X_train[:nTotalImages], 32)
print("filename","lable_real","lable_predict")
for i in range(nTotalImages):
    print(list_imagefilename[i],np.argmax(Y_train[i]),np.argmax(predict_result[i]))

'''
('filename', 'lable_real', 'lable_predict')
('subject01.leftlight', 0)
('subject02.leftlight', 1)
('subject03.leftlight', 2)
('subject04.leftlight', 3)
('subject05.leftlight', 4)
('subject06.leftlight', 5)
('subject07.leftlight', 6)
('subject08.leftlight', 7)
('subject09.leftlight', 8)
('subject10.leftlight', 9)
('subject11.leftlight', 10)
('subject12.leftlight', 11)
('subject13.leftlight', 12)
('subject14.leftlight', 13)
('subject15.leftlight', 14)
'''
