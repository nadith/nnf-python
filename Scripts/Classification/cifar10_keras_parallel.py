
from __future__ import print_function
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

import keras.layers
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import layer_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import math
from nnf.db.NNdb import NNdb
from nnf.core.callbacks.TensorBoardEx import TensorBoardEx
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.core.iters.ImageDataPreProcessor import ImageDataPreProcessor


batch_size = 32
num_classes = 10
epochs = 50
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

data_folder = r'F:\#DL_THEANO\Workspace\DL\DLN\DataFolder'
nndb_tr = NNdb.load(os.path.join(data_folder, "MATLAB_CIFAR10_tr_0_5000.mat"))
nndb_val = NNdb.load(os.path.join(data_folder, "MATLAB_CIFAR10_val_5000_6000.mat"))
x_train = nndb_tr.db_scipy
y_train = nndb_tr.cls_lbl
x_test = nndb_val.db_scipy
y_test = nndb_val.cls_lbl

# # The data, shuffled and split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# data_folder = r'F:\#DL_THEANO\Workspace\DL\DLN\DataFolder'
# db_dir = os.path.join(data_folder, "keras", "cifar-10-batches-py")
# (x_train2, y_train2), (x_test2, y_test2) = load_data(db_dir)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255



# Architecture
# INPUT -> CNN BLOCK1 -> CNN BLOCK2 -> ...
# AUX_INPUT -> AUX CNN BLOCK1 ^

main_input = Input(shape=x_train.shape[1:], name='main_input')

x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(main_input)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

y = Conv2D(32, (3, 3), activation='relu', padding='same', name='aux_block1_conv1')(main_input)
y = Conv2D(32, (3, 3), activation='relu', padding='same', name='aux_block1_conv2')(y)
y = MaxPooling2D((2, 2), strides=(2, 2), name='aux_block1_pool')(y)

auxiliary_input = Input(shape=x_train.shape[1:], name='auxiliary_input')
x = keras.layers.concatenate([x, y])


x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
auxiliary_output = Dense(num_classes, activation='softmax', name='aux_output')(x)

x = Dense(2048, activation='relu', name='fc2')(x)
main_output = Dense(num_classes, activation='softmax', name='main_output')(x)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


if not data_augmentation:
    print('Not using data augmentation.')

    import tensorflow as tf
    with tf.device('/gpu:1'):

        model.fit([x_train, x_train], [y_train, y_train],
                  epochs=epochs, batch_size=batch_size, validation_data=([x_test, x_test], [y_test, y_test]),
                  callbacks=[TensorBoardEx(log_dir='D:\\TF\\KERAS_PARALLEL', histogram_freq=True, write_images=True)])

        # model.fit([x_train , x_train], [y_train, y_train],
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           validation_data=(x_test, y_test),
        #           shuffle=True)
else:
    print('Using real-time data augmentation.')
    assert(False)

    # TODO: YET TO IMPLEMENT
    params = {}
    params['batch_size'] = batch_size
    params['class_mode'] = 'categorical'

    numpyIter = ImageDataPreProcessor().flow_ex(x_train, y_train, num_classes, params=params)


    # This will do preprocessing and realtime data augmentation:
    # datagen = ImageDataGenerator()
        # featurewise_center=False,  # set input mean to 0 over the dataset
        # samplewise_center=False,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=False,  # apply ZCA whitening
        # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        # horizontal_flip=True,  # randomly flip images
        # vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    import tensorflow as tf
    with tf.device('/gpu:1'):
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(numpyIter,
                            epochs=epochs,
                            steps_per_epoch=int(math.ceil(x_train.shape[0]/batch_size)), # nndb_tr.n),
                            validation_data=([x_test, x_test], [y_test, y_test]),
                            workers=4)

                            #callbacks=[TensorBoardEx(log_dir='D:\\TF\\KERAS', histogram_freq=True, write_images=True)])
