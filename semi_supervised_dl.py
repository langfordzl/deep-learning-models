#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:49:52 2018

@author: langfordz
"""

import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16


##############################################

NUM_CLASSES = 2
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 64
NEPOCH = 25
NTRAIN = 2680  # the number of training images
NVAL = 200  # the number of validation images
app = VGG16
weights = 'vgg16_model_custom.h5'

##############################################

plt.rcParams['figure.figsize'] = 10,10

def create_train(train_dir):
    datagen_train = image.ImageDataGenerator(featurewise_center=False,
                                             samplewise_center=False,
                                             featurewise_std_normalization=False,
                                             samplewise_std_normalization=False,
                                             zca_whitening=False,
                                             rotation_range=0,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True,
                                             vertical_flip=False,
                                             zoom_range=[.8, 1], 
                                             shear_range=0.3,
                                             channel_shift_range=30,
                                             fill_mode='reflect') 
    train_generator = datagen_train.flow_from_directory(
            train_dir,  # this is the target directory
            target_size=(HEIGHT, WIDTH),  # all images will be resized to 150x150
            batch_size=BATCH_SIZE,
            class_mode='categorical') 
    return train_generator 


def create_val(val_dir):
    datagen_val = image.ImageDataGenerator()
    validation_generator = datagen_val.flow_from_directory(
            val_dir,
            target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
    return validation_generator


itr = train_data = create_train('train')
Xtrain, Ytrain = itr.next()

itr = train_data = create_val('validation')
Xtest, Ytest = itr.next()

# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
##############################################
# Define the model

def getModel(WIDTH,HEIGHT,NUM_CLASSES):
    #Build keras model
    model=Sequential()
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(WIDTH,HEIGHT,3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))
    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # You must flatten the data for the dense layers
    model.add(Flatten())
    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # Output 
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    optimizer = SGD(lr=0.0001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = getModel(WIDTH,HEIGHT,NUM_CLASSES)
model.summary()

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NEPOCH, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_data=[Xtest, Ytest])


model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(Xtest, Ytest, verbose=2)
print('CV loss:', score[0])
print('CV accuracy:', score[1])

pt = model.predict(Xtest)
mse = (np.mean((pt-Ytest)**2))
print('CV MSE: ', mse)

##############################################
# Define the model


