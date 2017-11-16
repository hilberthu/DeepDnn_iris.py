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

X_train  = np.zeros((500,1,64,64))
Y_train  = np.zeros((500,15))
nTotalImages = 0
def loadImages():
    rootdir_image = "/home/alian/PycharmProjects/DeepDnn_iris.py/alian/yalefaces"
    list_paths = os.listdir(rootdir_image)
    image_index = 0
    for i in range(len(list_paths)):
        path = os.path.join(rootdir_image, list_paths[i])
        if (path.find("subject") < 0):
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
        Y_train[image_index] = np_utils.to_categorical(int(y)-1, num_classes=15)
        X_train[image_index][0] = image_pix/255
        image_index +=1
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
model = Sequential()

# Conv layer 1 output shape (32, 64, 64)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 64, 64),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 32, 32)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 32, 32)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 16, 16)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))


# Conv layer 2 output shape (32, 16, 16)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 8, 8)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))


# Fully connected layer 1 input shape (64 * 8 * 8) = (4096), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(15))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
X_train = X_train[:nTotalImages]
Y_train = Y_train[:nTotalImages]

X_test = X_train[:20]
Y_test = Y_train[:20]
model.fit(X_train, Y_train, epochs=20, batch_size=10,verbose=2)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, Y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('yale_model.h5')

'''
Epoch 1/20
29s - loss: 2.7419 - acc: 0.0843
Epoch 2/20
26s - loss: 2.5541 - acc: 0.2289
Epoch 3/20
27s - loss: 2.2768 - acc: 0.3554
Epoch 4/20
26s - loss: 1.8240 - acc: 0.5000
Epoch 5/20
25s - loss: 1.2757 - acc: 0.5964
Epoch 6/20
26s - loss: 0.8692 - acc: 0.7530
Epoch 7/20
26s - loss: 0.5652 - acc: 0.8554
Epoch 8/20
25s - loss: 0.4259 - acc: 0.8735
Epoch 9/20
28s - loss: 0.3139 - acc: 0.8916
Epoch 10/20
30s - loss: 0.2561 - acc: 0.9398
Epoch 11/20
28s - loss: 0.3536 - acc: 0.9157
Epoch 12/20
28s - loss: 0.1820 - acc: 0.9578
Epoch 13/20
28s - loss: 0.0953 - acc: 0.9940
Epoch 14/20
28s - loss: 0.0516 - acc: 0.9940
Epoch 15/20
28s - loss: 0.0423 - acc: 0.9940
Epoch 16/20
30s - loss: 0.0256 - acc: 1.0000
Epoch 17/20
28s - loss: 0.0243 - acc: 1.0000
Epoch 18/20
28s - loss: 0.0189 - acc: 1.0000
Epoch 19/20
29s - loss: 0.0116 - acc: 1.0000
Epoch 20/20
30s - loss: 0.0095 - acc: 1.0000

Testing ------------
20/20 [==============================] - 3s
('\ntest loss: ', 0.0060974806547164917)
('\ntest accuracy: ', 1.0)
'''