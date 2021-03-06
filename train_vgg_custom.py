'''
Training using the VGG network only
We are using the network for training only and not transfer learning/fine-tuning
'''
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.layers.core import Flatten
from keras.optimizers import SGD
from sys import argv

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from os import listdir
from os.path import join
##############################################
# taken from https://github.com/stratospark/food-101-keras
import matplotlib.image as img
from scipy.misc import imresize
import collections
from keras.callbacks import ModelCheckpoint
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

def create_data(train_dir, val_dir):
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
    datagen_val = image.ImageDataGenerator()
    train_generator = datagen_train.flow_from_directory(
            train_dir,  # this is the target directory
            target_size=(HEIGHT, WIDTH),  # all images will be resized to 150x150
            batch_size=BATCH_SIZE,
            class_mode='categorical') 
    validation_generator = datagen_val.flow_from_directory(
            val_dir,
            target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
    return train_generator, validation_generator 

train_generator, validation_generator = create_data('train', 'validation')

def reverse_preprocess_input(x0):
    x = x0 / 2.0
    x += 0.5
    x *= 255.
    return x

def show_images(test_generator, unprocess=True):
    for x in test_generator:
        fig, axes = plt.subplots(nrows=8, ncols=4)
        fig.set_size_inches(8, 8)
        page = 0
        page_size = 32
        start_i = page * page_size
        for i, ax in enumerate(axes.flat):
            img = x[0][i+start_i]
            if unprocess:
                im = ax.imshow( reverse_preprocess_input(img).astype('uint8') )
            else:
                im = ax.imshow(img)
            ax.set_axis_off()
            ax.title.set_visible(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.subplots_adjust(left=0, wspace=0, hspace=0)
        plt.show()
        break

#show_images(unprocess=False)

def train_keras_model(nepochs, app, NTRAIN, NVAL, train_generator, test_generator, BATCH_SIZE):
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
    checkpointer = ModelCheckpoint(filepath='model4.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    try:
        history = model.fit_generator(
            train_generator, 
            steps_per_epoch=NTRAIN // BATCH_SIZE,
            epochs=nepochs,
            validation_data=validation_generator,
            validation_steps=NVAL // BATCH_SIZE,
            callbacks=[checkpointer])
    except KeyboardInterrupt as e:
        print('Got keyboard interrupt. Ending training.')    
    #print('Saving model to h5 file!')
    #model.save(weights)
    return history 

history = train_keras_model(NEPOCH, app, NTRAIN, NVAL, train_generator, validation_generator, BATCH_SIZE)

def plot_training(history, acc, loss):
    ## summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc)
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss)
    plt.close()

plot_training(history, 'model_accuracy.pdf', 'model_loss.pdf')


#def run_test(weights, img, WIDTH, HEIGHT):
    ## load model 
    #model = load_model(weights)
    #img = Image.open(img)
    #img = img.resize((WIDTH, HEIGHT))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    ## remove alpha channel if present
    #x = x[:,:,:,:3]
    ## rescale data 
    ##x = x * 1. / 255 
    #preds = model.predict(x)
    #pred = np.argmax(preds)
    #prob = preds[0][pred]
    #print('Predicted %s with probability %.2f' % (pred, prob))
    #return pred
    

#print ('Load test image!')
#img = ('test/f-15-eagles-from-the-67th-fighter-squadron-perform-an-elephant-walk-j13abm.jpg')    
#pred = run_test(weights, img, WIDTH, HEIGHT)