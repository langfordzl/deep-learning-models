'''
Training using the VGG network only
We are using the network for training only and not transfer learning/fine-tuning
'''
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.layers.core import Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from sys import argv
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
##############################################
# taken from https://github.com/stratospark/food-101-keras
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

import h5py
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
##############################################

NUM_CLASSES = 2
WIDTH = 550
HEIGHT = 550
BATCH_SIZE = 64
NEPOCH = 5
NTRAIN = 2680  # the number of training images
NVAL = 200  # the number of validation images
app = VGG16
weights = 'vgg16_model_custom.h5'


import multiprocessing as mp
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
def load_images(root, min_side=224):
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

X_train, y_train = load_images('train', min_side=224)
X_val, y_val = load_images('validation', min_side=224)

def create_data(dir, num):
    datagen = image.ImageDataGenerator() 
    generator = datagen.flow_from_directory(
            dir,  # this is the target directory
            target_size=(HEIGHT, WIDTH),  # all images will be resized 
            batch_size=num,
            class_mode='categorical') 
    return generator

itr = train_generator = create_data('train', NTRAIN)
X, y = itr.next()
    
    
print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_val shape', X_val.shape)
print('y_val shape', y_val.shape)

# View test image 
#a = X[0]*255
#b = Image.fromarray(a.astype('uint8'), 'RGB')
    

# train model
def train_model(nepochs, app, weights, X_train, y_train, X_val, y_val):
    print('Initializing a fresh model to train.')
    input_tensor = Input(shape=(HEIGHT, WIDTH, 3))  # tf ordering, i.e., height, width, channels
    base_model = app(input_tensor=input_tensor, include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x) 
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    # this is the model we will train
    model = Model(input=input_tensor, output=predictions)
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    # print a summary of the model and then start training
    model.summary()
    history = model.fit(
        X_train, y_train, 
        epochs=nepochs,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE
    )
    print('Saving model to h5 file!')
    model.save(weights)
    return history 

def plot_training(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('test_accuracy.pdf')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('test_loss.pdf')
    plt.close()

def run_test(weights, img, WIDTH, HEIGHT):
    # load model 
    model = load_model(weights)
    img = Image.open(img)
    img = img.resize((WIDTH, HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # remove alpha channel if present
    x = x[:,:,:,:3]
    # rescale data 
    #x = x * 1. / 255 
    preds = model.predict(x)
    pred = np.argmax(preds)
    prob = preds[0][pred]
    print('Predicted %s with probability %.2f' % (pred, prob))
    return pred
    
if __name__ == '__main__':
    #train_generator, validation_generator = create_data('train', 'validation')
    history = train_model(NEPOCH, app, weights, X_train, y_train, X_val, y_val)
    #plot_training(history)
    #print ('Done training!')
    # run test 
    print ('Load test image!')
    img = ('test/f-15-eagles-from-the-67th-fighter-squadron-perform-an-elephant-walk-j13abm.jpg')    
    pred = run_test(weights, img, WIDTH, HEIGHT)