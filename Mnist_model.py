#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:17:46 2019

@author: hakunamatata
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.utils import to_categorical

train_im = pd.read_csv('train.csv')
train_im.head()
train_label = train_im['label']
train_im.drop(['label'], axis = 1, inplace=True)
train_label = to_categorical(train_label)
def data_preprocesing(data):
    data = np.array(data).reshape(len(data), 28,28, 1)
    data = data.astype('float32')/255
    return data

train_im = data_preprocesing(train_im)
# Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

# Training
num_epochs = 30
batch_size = 800

model.fit(train_im, train_label, epochs = num_epochs, batch_size = batch_size)


# Getting Test Results

test = pd.read_csv('test.csv')
test = data_preprocesing(test)
prediction = model.predict(test)
pred = []
for i in range(prediction.shape[0]):
    pred.append(np.argmax(prediction[i,:]))
result = pd.DataFrame({'ImageId' : range(0, prediction.shape[0]), 'Label':pred})
result.to_csv('result.csv', Index = False)