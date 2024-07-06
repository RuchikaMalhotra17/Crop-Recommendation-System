import tensorflow as tf
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *

# Define the paths to the training, validation, and test directories
train_path = 'C:/Users/Rishika Malhotra/AppData/Local/Programs/Python/Python39/sih/archive/Train/Train'
val_path = 'C:/Users/Rishika Malhotra/AppData/Local/Programs/Python/Python39/sih/archive/Test/Test'
test_path = 'C:/Users/Rishika Malhotra/AppData/Local/Programs/Python/Python39/sih/archive/Validation/Validation'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(360, 360),
        batch_size=32)
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(360, 360),
        batch_size=32)

validation_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=(360, 360),
        batch_size=32)
efficient = tf.keras.applications.mobilenet.MobileNet(
    include_top=False, weights='imagenet',
    input_shape=(360, 360, 3), pooling='max', classes=3,
    classifier_activation='relu')
for layer in efficient.layers:
    layer.trainable = False
    model = tf.keras.models.Sequential()
model.add(efficient)
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(train_generator, validation_data=validation_generator, epochs=10)
model.save("Disease_Detection.h5")
print("Model saved")
