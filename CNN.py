#mount drive
from google.colab import drive
drive.mount('/content/drive')

#Libraries
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

#data path
data_path = '/content/drive/MyDrive/Colab Notebooks/Hands_dataset'
img_rows = 100
img_cols = 100
batch_size = 64

#Image Data Generator
def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

#Image Generator 
train_datagen = ImageDataGenerator(preprocessing_function=prep_fn,                                  
                                  #rescale=1. / 255,                               
                                  validation_split=0.3) # set validation split                                                                              

train_generator = train_datagen.flow_from_directory(data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory( data_path,
                                                          target_size=(img_rows, img_cols),
                                                          batch_size=batch_size,
                                                          class_mode='categorical',
                                                          subset='validation')


# Build model CNN
num_classes = 10

model = Sequential()
model.add(Convolution2D(64, (5, 5), input_shape=(img_rows, img_cols, 3), padding='valid'))
model.add(layers.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(32, (5, 5), padding='valid'))
model.add(layers.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(16, (5, 5), padding='valid'))
model.add(layers.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#compile
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Train
num_of_train_samples = 1448
num_of_test_samples = 614  
epochs = 50

history = model.fit(train_generator,
                    #generator3,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size)

