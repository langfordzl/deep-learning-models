#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:46:55 2018

@author: langfordz
Transfer Learning with Syn. Datasets
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
from os import listdir
from os.path import join
import multiprocessing as mp
import matplotlib.image as img
from scipy.misc import imresize
import matplotlib.pyplot as plt
import random 
from PIL import Image 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
num_processes = 6
pool = mp.Pool(processes=num_processes)


# create datasets
class_to_ix = {}
ix_to_class = {}
with open('classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    class_to_ix = dict(zip(classes, range(len(classes))))
    ix_to_class = dict(zip(range(len(classes)), classes))
    class_to_ix = {v: k for k, v in ix_to_class.items()}
    
sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))

# Load dataset images and resize to meet minimum width and height pixel size
def load_images(root, min_side=550):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    for i, subdir in enumerate(listdir(root)):
        imgs = listdir(join(root, subdir))
        class_ix = class_to_ix[subdir]
        print(i, class_ix, subdir)
        for img_name in imgs:
            img_arr = img.imread(join(root, subdir, img_name))
            img_arr_rs = img_arr
            try:
                w, h, _ = img_arr.shape
                if w > min_side:
                    #wpercent = (min_side/float(w))
                    #hsize = int((float(h)*float(wpercent)))
                    #print('new dims:', min_side, hsize)
                    img_arr_rs = imresize(img_arr, (min_side, min_side))
                    resize_count += 1
                elif h > min_side:
                    #hpercent = (min_side/float(h))
                    #wsize = int((float(w)*float(hpercent)))
                    #print('new dims:', wsize, min_side)
                    img_arr_rs = imresize(img_arr, (min_side, min_side))
                    resize_count += 1
                all_imgs.append(img_arr_rs)
                all_classes.append(class_ix)
            except:
                print('Skipping bad image: ', subdir, img_name)
                invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)


##############################################
# Load datasets

xtrain, ytrain = load_images('train', min_side=224)
xval, yval = load_images('validation', min_side=224)

def view_image(x, y):
    num = random.randrange(0, 2680, 2)
    im = x[num]
    lab = y[num]
    print ("Class =", lab)
    im = im
    image = Image.fromarray(im.astype('uint8'), 'RGB')
    image.show()

#view_image(xtrain, ytrain)


##############################################
# Image Augmentation
import cv2
from sklearn.model_selection import StratifiedShuffleSplit, KFold

def get_augment(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []   
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    more_images = np.concatenate((imgs,v,h))
    return more_images

# Stratified splitting
    
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2)

for train_index, cv_index in sss.split(xtrain, ytrain):
    X_train, X_cv = xtrain[train_index], xtrain[cv_index]
    y_train, y_cv = ytrain[train_index], ytrain[cv_index]
    Xtr_more = get_augment(X_train) 
    Xcv_more = get_augment(X_cv) 
    Ytr_more = np.concatenate((y_train,y_train,y_train))
    Ycv_more = np.concatenate((y_cv,y_cv,y_cv))    

print (Xtr_more.shape)
print (Xcv_more.shape)
print (Ytr_more.shape)
print (Ycv_more.shape)


       
##############################################
# CNN Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.models import Model

from keras.optimizers import Adam
from keras.optimizers import rmsprop
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from keras.applications.vgg19 import VGG19

# Finally create generator
def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=10, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def getVggModel(HEIGHT, WIDTH, NUM_CLASSES):
    input_tensor = Input(shape=(HEIGHT, WIDTH, 3))  # tf ordering, i.e., height, width, channels
    base_model = VGG16(weights='imagenet', input_tensor=input_tensor, include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x) 
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    # this is the model we will train
    model = Model(input=input_tensor, output=predictions)
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # print a summary of the model and then start training
    model.summary()
    return model


def getVggModel2(HEIGHT, WIDTH, NUM_CLASSES):
    input_tensor = Input(shape=(HEIGHT, WIDTH, 3))  # tf ordering, i.e., height, width, channels
    base_model = VGG16(input_tensor=input_tensor, include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x) 
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    # this is the model we will train
    model = Model(input=input_tensor, output=predictions)
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # print a summary of the model and then start training
    model.summary()
    return model


HEIGHT, WIDTH = 224, 224
NUM_CLASSES = 2

##############################################
# Test Runs

model = getVggModel(HEIGHT, WIDTH, NUM_CLASSES)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.vgg16_wts_tl.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model.fit(Xtr_more, Ytr_more, batch_size=16, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_data=(Xcv_more, Ycv_more))
#
#
model = getVggModel2(HEIGHT, WIDTH, NUM_CLASSES)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.vgg16_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model.fit(Xtr_more, Ytr_more, batch_size=16, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_data=(Xcv_more, Ycv_more))