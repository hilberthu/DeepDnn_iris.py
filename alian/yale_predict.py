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
nTotalImages = 0
def loadImages():
    rootdir_image = "/home/alian/PycharmProjects/DeepDnn_iris.py/alian/yalefaces"
    global list_paths
    list_paths = os.listdir(rootdir_image)
    for i in range(len(list_paths)):
        path = os.path.join(rootdir_image, list_paths[i])
        if path.find("subject") < 0:
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
        Y_train[i] = np_utils.to_categorical(int(y)-1, num_classes=15)
        X_train[i][0] = image_pix/255
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
print(list_paths)
model = load_model("yale_model.h5")
predict_result = model.predict(X_train, 166)
for i in range(166):
    print(list_paths[i],np.argmax(Y_train[i]))

'''
('subject01.gif', 0)
('subject01.glasses', 0)
('subject01.glasses.gif', 0)
('subject01.happy', 0)
('subject01.leftlight', 0)
('subject01.noglasses', 0)
('subject01.normal', 0)
('subject01.rightlight', 0)
('subject01.sad', 0)
('subject01.sleepy', 0)
('subject01.surprised', 0)
('subject01.wink', 0)
('subject02.centerlight', 1)
('subject02.glasses', 1)
('subject02.happy', 1)
('subject02.leftlight', 1)
('subject02.noglasses', 1)
('subject02.normal', 1)
('subject02.rightlight', 1)
('subject02.sad', 1)
('subject02.sleepy', 1)
('subject02.surprised', 1)
('subject02.wink', 1)
('subject03.centerlight', 2)
('subject03.glasses', 2)
('subject03.happy', 2)
('subject03.leftlight', 2)
('subject03.noglasses', 2)
('subject03.normal', 2)
('subject03.rightlight', 2)
('subject03.sad', 2)
('subject03.sleepy', 2)
('subject03.surprised', 2)
('subject03.wink', 2)
('subject04.centerlight', 3)
('subject04.glasses', 3)
('subject04.happy', 3)
('subject04.leftlight', 3)
('subject04.noglasses', 3)
('subject04.normal', 3)
('subject04.rightlight', 3)
('subject04.sad', 3)
('subject04.sleepy', 3)
('subject04.surprised', 3)
('subject04.wink', 3)
('subject05.centerlight', 4)
('subject05.glasses', 4)
('subject05.happy', 4)
('subject05.leftlight', 4)
('subject05.noglasses', 4)
('subject05.normal', 4)
('subject05.rightlight', 4)
('subject05.sad', 4)
('subject05.sleepy', 4)
('subject05.surprised', 4)
('subject05.wink', 4)
('subject06.centerlight', 5)
('subject06.glasses', 5)
('subject06.happy', 5)
('subject06.leftlight', 5)
('subject06.noglasses', 5)
('subject06.normal', 5)
('subject06.rightlight', 5)
('subject06.sad', 5)
('subject06.sleepy', 5)
('subject06.surprised', 5)
('subject06.wink', 5)
('subject07.centerlight', 6)
('subject07.glasses', 6)
('subject07.happy', 6)
('subject07.leftlight', 6)
('subject07.noglasses', 6)
('subject07.normal', 6)
('subject07.rightlight', 6)
('subject07.sad', 6)
('subject07.sleepy', 6)
('subject07.surprised', 6)
('subject07.wink', 6)
('subject08.centerlight', 7)
('subject08.glasses', 7)
('subject08.happy', 7)
('subject08.leftlight', 7)
('subject08.noglasses', 7)
('subject08.normal', 7)
('subject08.rightlight', 7)
('subject08.sad', 7)
('subject08.sleepy', 7)
('subject08.surprised', 7)
('subject08.wink', 7)
('subject09.centerlight', 8)
('subject09.glasses', 8)
('subject09.happy', 8)
('subject09.leftlight', 8)
('subject09.noglasses', 8)
('subject09.normal', 8)
('subject09.rightlight', 8)
('subject09.sad', 8)
('subject09.sleepy', 8)
('subject09.surprised', 8)
('subject09.wink', 8)
('subject10.centerlight', 9)
('subject10.glasses', 9)
('subject10.happy', 9)
('subject10.leftlight', 9)
('subject10.noglasses', 9)
('subject10.normal', 9)
('subject10.rightlight', 9)
('subject10.sad', 9)
('subject10.sleepy', 9)
('subject10.surprised', 9)
('subject10.wink', 9)
('subject11.centerlight', 10)
('subject11.glasses', 10)
('subject11.happy', 10)
('subject11.leftlight', 10)
('subject11.noglasses', 10)
('subject11.normal', 10)
('subject11.rightlight', 10)
('subject11.sad', 10)
('subject11.sleepy', 10)
('subject11.surprised', 10)
('subject11.wink', 10)
('subject12.centerlight', 11)
('subject12.glasses', 11)
('subject12.happy', 11)
('subject12.leftlight', 11)
('subject12.noglasses', 11)
('subject12.normal', 11)
('subject12.rightlight', 11)
('subject12.sad', 11)
('subject12.sleepy', 11)
('subject12.surprised', 11)
('subject12.wink', 11)
('subject13.centerlight', 12)
('subject13.glasses', 12)
('subject13.happy', 12)
('subject13.leftlight', 12)
('subject13.noglasses', 12)
('subject13.normal', 12)
('subject13.rightlight', 12)
('subject13.sad', 12)
('subject13.sleepy', 12)
('subject13.surprised', 12)
('subject13.wink', 12)
('subject14.centerlight', 13)
('subject14.glasses', 13)
('subject14.happy', 13)
('subject14.leftlight', 13)
('subject14.noglasses', 13)
('subject14.normal', 13)
('subject14.rightlight', 13)
('subject14.sad', 13)
('subject14.sleepy', 13)
('subject14.surprised', 13)
('subject14.wink', 13)
('subject15.centerlight', 14)
('subject15.glasses', 14)
('subject15.happy', 14)
('subject15.leftlight', 14)
('subject15.noglasses', 14)
('subject15.normal', 14)
('subject15.rightlight', 14)
('subject15.sad', 14)
('subject15.sleepy', 14)
('subject15.surprised', 14)
'''
