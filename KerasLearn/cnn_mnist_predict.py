# -*- coding: UTF-8 -*-
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.image as mping
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt
model = load_model("mnist_model.h5")


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

toPredict = X_test[0:10]
predict_result = model.predict(toPredict,32)

index = 0

for oneResult in predict_result:
    print(np.argmax(oneResult),np.argmax(y_test[index]))
    index += 1

toPredictImage = mping.imread("images/2.png")

wolf_gray=(toPredictImage[:,:,1])

#对图像进行缩放 ，变成28*28
wolf_newsize =  tf.image.resize_images(toPredictImage, [28, 28], method=tf.image.ResizeMethod.BILINEAR)

wolf_newsize = wolf_newsize[:,:,1]
sess=tf.Session()
topredictList1 = np.random.rand(1,1,28,28)
with sess:
    newImage = wolf_newsize.eval()

topredictList1[0][0] = newImage
for i in range(28):
    for j in range(28):
        if(topredictList1[0][0][i][j] > 0.9):
            topredictList1[0][0][i][j] = 0
        else:
            topredictList1[0][0][i][j] = 1

plt.imshow(X_test[0][0])
plt.show()
plt.imshow(topredictList1[0][0])
plt.show()
predict_result = model.predict(topredictList1, 32)
print(np.argmax(predict_result[0]))
print("alian")
#loss, accuracy = model.evaluate(X_test, y_test)

#print('\ntest loss: ', loss)
#print('\ntest accuracy: ', accuracy)