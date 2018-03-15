
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from sys import argv
import matplotlib.pyplot as plt

########################

NUM_CLASSES = 2
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 32
NEPOCH = 10
NTRAIN = 2680  # the number of training images
NVAL = 200  # the number of validation images

def load_dataset(index_file):
    imagefiles = []
    classes = []

    with open(index_file, 'r') as f:
        for line in f.readlines():
            line = line.split()
            imagefiles.append(line[0])
            classes.append(' '.join(line[1:]))
    return imagefiles, classes

def train_model(train_dir, test_dir, nepochs, apps):

    datagen = image.ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        vertical_flip=True,
        zoom_range=0.3, 
        shear_range=0.3,
        fill_mode='nearest')
    
    #datagen = image.ImageDataGenerator(
        #featurewise_center=False,
        #samplewise_center=False,
        #featurewise_std_normalization=False,
        #samplewise_std_normalization=False,
        #zca_whitening=False,
        #zca_epsilon=1e-6,
        #rotation_range=0.,
        #width_shift_range=0.,
        #height_shift_range=0.,
        #shear_range=0.,
        #zoom_range=0.,
        #channel_shift_range=0.,
        #fill_mode='nearest',
        #cval=0.,
        #horizontal_flip=False,
        #vertical_flip=False,
        #rescale=1./255,
        #preprocessing_function=None)
    
    datagen_val = image.ImageDataGenerator()

    print('Initializing a fresh model to train.')
    input_tensor = Input(shape=(HEIGHT, WIDTH, 3))  # tf ordering, i.e., height, width, channels
    print (apps)
    base_model = apps(input_tensor=input_tensor, include_top=False)
    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    # this is the model we will train
    model = Model(input=input_tensor, output=predictions)


    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.summary()
    try:
        history = model.fit_generator(datagen.flow_from_directory(train_dir, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE,
                                                        class_mode='categorical'),
                steps_per_epoch=NTRAIN//BATCH_SIZE,
                epochs=nepochs,
                validation_data=datagen_val.flow_from_directory(test_dir, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE,
                class_mode='categorical'),
                validation_steps=NVAL//BATCH_SIZE,
                )
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(apps + 'test_accuracy.pdf')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(apps + 'test_loss.pdf')
        plt.close()
        
    except KeyboardInterrupt as e:
        print('Got keyboard interrupt. Ending training.')
    
    print('Saving model to modelweights.h5')
    model.save(apps + '_aug1.h5')


if __name__ == '__main__':
    # list of deep learning models extracted from Keras 
    # https://keras.io/applications/
    train_model('train/', 'validation/', NEPOCH, VGG16)    
    
    
