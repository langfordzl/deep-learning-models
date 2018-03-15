#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:23:52 2018

@author: langfordz
"""

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image 

xtrain=np.load('trainimages.npy')
ytrain=np.load('traintargets.npy')
xval=np.load('valimages.npy')
yval=np.load('valtargets.npy')
    
# loop through and create list of augmented images

#Apply various augmentations from imgaug
def blur():
    blurer=[iaa.GaussianBlur((0, 3.0)),iaa.AverageBlur(k=(2, 7)),iaa.MedianBlur(k=(3, 11))]
    return blurer[np.random.randint(0,3)]

def dropout():
    dropper=[iaa.Dropout((0.01, 0.1), per_channel=0.5), iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)]
    return dropper[np.random.randint(0,2)]

def noise():
    return iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)

def greyscale():
    return iaa.Grayscale(alpha=(0.0, 1.0))

def invert():
    return iaa.Invert(0.05, per_channel=True)

def hue():
    return iaa.AddToHueAndSaturation((-20, 20))


def sharpen(): 
    return iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))

def emboss(): 
    return iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))


# Augmentations to apply 
def scale(): 
    return iaa.Affine(scale=(0.5, 1.5))

def rotate(): 
    return iaa.Affine(rotate=(-45, 45))

def add(): 
    return iaa.Add((-10, 10), per_channel=0.5)

def multiply():
    return iaa.Multiply((0.5, 1.5), per_channel=0.5)

def contrast():
    return iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)



# apply augmentation over single image
def augment(imgs):
    augmented=[]
    #augs=[blur(),dropout(),noise(),contrast(),greyscale(),invert(),hue(),add(),multiply(),sharpen(),emboss()]
    augs=[scale(), rotate(), add(), multiply(), contrast()]
    #augs=[scale()]
    for aug in augs:
        #print ("Working on:", aug)
        seq=iaa.Sequential(aug)
        aug_img=seq.augment_image(imgs)
        augmented.append(aug_img)
    return np.asarray(augmented)

# apply augmentation to a list of images 
def aug_images(imgs):
    images=[]
    for img in imgs:
        print ("Working on image with shape:", img.shape)
        aug_img = augment(img)
        images.append(aug_img)
    images = np.asarray(images)
    print (images.shape)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])
    return images



aug = aug_images(xtrain)
#images = np.asarray(aug)
#images = images.reshape(images.shape[0]*images.shape[1], 512, 512, 3)
# create labels based on the number of augmentations
augs=[scale(), rotate(), add(), multiply(), contrast()]
y = np.tile(ytrain,(len(augs),1))
y = y.reshape(y.shape[0]*y.shape[1])
    
# save vectors

np.save('xtrain_aug_512.npy', aug)
np.save('ytrain_aug_512.npy', y)




