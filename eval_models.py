#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:53:24 2018

@author: langfordz
Model Evaluation
"""

from keras.models import Sequential
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16

from keras.layers import Dense, Input, Flatten
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np
from PIL import Image 
import collections
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy.misc import imresize
from os import listdir
from os.path import join

# Models
# inceptionv3_2classes.h5  resnet_model1.h5  resnet_model2.h5  vgg16_model.h5  vgg16_model_tl.h5
model = load_model('inceptionv3_2classes_mixed.h5')
model.summary()

def load_img(img, WIDTH, HEIGHT):
    ## load model 
    img = Image.open(img)
    img = img.resize((WIDTH, HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # remove alpha channel if present
    x = x[:,:,:,:3]
    # rescale data 
    #x = x * 1. / 255 
    return x

class_to_ix = {}
ix_to_class = {}
with open('classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    class_to_ix = dict(zip(classes, range(len(classes))))
    ix_to_class = dict(zip(range(len(classes)), classes))
    class_to_ix = {v: k for k, v in ix_to_class.items()}
    
sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))

def load_images(root, min_side):
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
            img_arr_rs = imresize(img_arr, (min_side, min_side))
            resize_count += 1
            all_imgs.append(img_arr_rs)
            all_classes.append(class_ix)
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)


X_test, y_test = load_images('data/test', 299)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print (y_pred)


# hard coded to plot 20 images! 
def show_images_prediction(plot):
    page = 0
    page_size = 10
    nrows = 4
    ncols = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fig.set_size_inches(8, 8)
    #fig.tight_layout()
    #imgs = np.random.choice((y_all == n_class).nonzero()[0], nrows * ncols)
    start_i = page * page_size
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(X_test[i+start_i])
        ax.set_axis_off()
        ax.title.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        predicted = ix_to_class[y_pred[i+start_i]]
        match = predicted ==  ix_to_class[y_test[start_i + i]]
        ec = (1, .5, .5)
        fc = (1, .8, .8)
        if match:
            ec = (0, .6, .1)
            fc = (0, .7, .2)
        # predicted label
        ax.text(0, 240, 'P: ' + predicted, size=10, rotation=0,
            ha="left", va="top",
             bbox=dict(boxstyle="round",
                   ec=ec,
                   fc=fc,
                   )
             )
        #if not match:
            # true label
        #    ax.text(0, 300, 'A: ' + ix_to_class[y_test[start_i + i]], size=10, rotation=0,
        #        ha="left", va="top",
        #         bbox=dict(boxstyle="round",
        #               ec=ec,
        #               fc=fc,
        #               )
        #         )
    plt.subplots_adjust(left=0, wspace=1, hspace=0)
    plt.tight_layout()
    plt.savefig(plot)
    plt.close() 

from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

show_images_prediction('inceptionv3_2classes_mixed2.pdf')


cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
class_names = [ix_to_class[i] for i in range(2)]


plt.figure()
fig = plt.gcf()
fig.set_size_inches(32, 32)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization',
                      cmap=plt.cm.cool)
plt.savefig('inceptionv3_confusionmatrix_mixed2.pdf')

plt.close() 











