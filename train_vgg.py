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

NUM_CLASSES = 2
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 64
NEPOCH = 50
NTRAIN = 2680  # the number of training images
NVAL = 200  # the number of validation images
app = VGG16
weights = 'vgg16_model2.h5'

# create datasets
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
    
    

# train model
def train_model(nepochs, app, weights, train_generator, validation_generator):
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
    try:
        history = model.fit_generator(
            train_generator, 
            steps_per_epoch=NTRAIN // BATCH_SIZE,
            epochs=nepochs,
            validation_data=validation_generator,
            validation_steps=NVAL // BATCH_SIZE)
    except KeyboardInterrupt as e:
        print('Got keyboard interrupt. Ending training.')
    
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
    plt.savefig('test_accuracy2.pdf')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('test_loss2.pdf')
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
    train_generator, validation_generator = create_data('train', 'validation')
    history = train_model(NEPOCH, app, weights, train_generator, validation_generator)
    plot_training(history)
    print ('Done training!')
    # run test 
    print ('Load test image!')
    img = ('test/f-15-eagles-from-the-67th-fighter-squadron-perform-an-elephant-walk-j13abm.jpg')    
    #pred = run_test(weights, img, WIDTH, HEIGHT)