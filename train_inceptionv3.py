'''
Training using the InceptionV3 network only
We are using the network for training only and not transfer learning/fine-tuning
'''
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, Input, AveragePooling2D
from keras.layers.core import Flatten, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
##############################################
# taken from https://github.com/stratospark/food-101-keras

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

##############################################

NUM_CLASSES = 2
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 32
NEPOCH = 25
NTRAIN = 2680  # the number of training images
NVAL = 200  # the number of validation images
app = InceptionV3
weights = 'inceptionv3_2classes.h5'

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

def train_keras_model(nepochs, app, weights, NTRAIN, NVAL, train_generator, test_generator, BATCH_SIZE):
    print('Initializing a fresh model to train.')
    input_tensor = Input(shape=(HEIGHT, WIDTH, 3))  # tf ordering, i.e., height, width, channels
    base_model = app(input_tensor=input_tensor, include_top=False)
    x = base_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    # this is the model we will train
    model = Model(input=input_tensor, output=predictions)
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    # print a summary of the model and then start training
    model.summary()
    checkpointer = ModelCheckpoint(filepath='inceptionv3_2classes_weights.hdf5', verbose=1, save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    try:
        history = model.fit_generator(
            train_generator, 
            steps_per_epoch=NTRAIN // BATCH_SIZE,
            epochs=nepochs,
            validation_data=validation_generator,
            validation_steps=NVAL // BATCH_SIZE,
            callbacks=[checkpointer, earlyStopping, reduce_lr_loss])
    except KeyboardInterrupt as e:
        print('Got keyboard interrupt. Ending training.')    
    print('Saving model to h5 file!')
    model.save(weights)
    return history 

history = train_keras_model(NEPOCH, app, weights, NTRAIN, NVAL, train_generator, validation_generator, BATCH_SIZE)

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

plot_training(history, 'InceptionV3model_accuracy.pdf', 'InceptionV3model_loss.pdf')


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