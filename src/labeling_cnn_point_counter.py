import os
import copy
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot
from matplotlib import image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras_preprocessing.image import ImageDataGenerator

import csv_handler

from constants import IMAGES_PATH, TRAIN_WIDTH, TRAIN_HEIGHT, HIDDEN_LAYER_ACTIVATION, \
    OUTPUT_NEURON_NUMBER, OUTPUT_LAYER_ACTIVATION, METRICS, EPOCHS, BATCH_SIZE
from csv_handler import COLUMN_NAMES

# const variables

MODEL_FILE = "model.txt"

_TRAIN_PCT = 0.9

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Preloading...")



# load the images, and labels

image_names = []
images = []
labels = []

for picture in os.listdir():
    image_names.append(picture)
    img = cv2.imread(picture, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img_expanded = img[:, :, np.newaxis]
    images.append(img_expanded)
    labels.append(int(picture.split(".")[0].split("_")[-1]))

# normalizing the labels

min_v = 2
max_v = 12

for i in range(len(labels)):
    labels[i] = (labels[i] - min_v) / (max_v - min_v)

# cut the data to train, validation, and test sets

train_cut = int(len(images) * _TRAIN_PCT)

train_images = images[0:train_cut]
test_images = images[train_cut:]

train_labels = labels[0:train_cut]
test_labels = labels[train_cut:]

image_names = image_names[train_cut:]

train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Creating the model...")

model = keras.Sequential([
    layers.Dropout(0.3, input_shape=(TRAIN_WIDTH, TRAIN_HEIGHT, 1)),
    layers.AveragePooling2D(2, 2),
    layers.Conv2D(32, 3, activation=HIDDEN_LAYER_ACTIVATION,
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5)),
    layers.AveragePooling2D(2, 2),
    layers.Dropout(0.3),
    layers.Conv2D(16, 3, activation=HIDDEN_LAYER_ACTIVATION,
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5)),
    layers.AveragePooling2D(2, 2),
    layers.Dropout(0.3),
    layers.Conv2D(8, 3, activation=HIDDEN_LAYER_ACTIVATION,
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation=HIDDEN_LAYER_ACTIVATION,
                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                 bias_regularizer=regularizers.l2(1e-4),
                 activity_regularizer=regularizers.l2(1e-5)),
    layers.Dropout(0.3),
    layers.Dense(512, activation=HIDDEN_LAYER_ACTIVATION,
                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                 bias_regularizer=regularizers.l2(1e-4),
                 activity_regularizer=regularizers.l2(1e-5)),
    layers.Dense(1, activation=OUTPUT_LAYER_ACTIVATION)
])

# train on the model

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Compiling model")

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError())

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Starting the learning")

history = model.fit(
    x=train_images,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode="min", restore_best_weights=True)])

images = images[train_cut:]


for i in range(len(image_names)):
    print(image_names[i])
    print(int(model.predict(np.expand_dims(test_images[i], (0, 3)))[0][0] * (max_v - min_v)) + min_v)
    print("--------------------------------")
