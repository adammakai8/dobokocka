import os
import random
import copy
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator
import h5py

import csv_handler

# const variables

from constants import IMAGES_PATH, TRAIN_WIDTH, TRAIN_HEIGHT, HIDDEN_LAYER_ACTIVATION, \
    OUTPUT_NEURON_NUMBER, OUTPUT_LAYER_ACTIVATION, METRICS, EPOCHS, BATCH_SIZE
from csv_handler import COLUMN_NAMES

_TRAIN_PCT = 0.8
_VALIDATION_PCT = 0.1

# variables

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Preloading...")

os.chdir(IMAGES_PATH)

# create the csv

csv_handler.create_csv()

# create image generator

df = pd.read_csv(csv_handler.CSV_FILE_NAME, sep=",")
train_cut = int(len(df) * _TRAIN_PCT)
validation_cut = int(len(df) * _VALIDATION_PCT)

df_train = df[0:train_cut]
df_validation = df[train_cut:(train_cut+validation_cut)]
df_test = df[(train_cut+validation_cut):]

column_names_without_filename_and_id = copy.deepcopy(COLUMN_NAMES)
column_names_without_filename_and_id.pop()
column_names_without_filename_and_id.pop(0)

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Creating the generator...")

training_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = training_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=os.getcwd()+"\\\\"+csv_handler.FROM_DATA_TO_PREPROCESSED_PATH,
    x_col=COLUMN_NAMES[-1],
    y_col=column_names_without_filename_and_id,
    target_size=(TRAIN_WIDTH, TRAIN_HEIGHT),
    color_mode="grayscale",
    class_mode="other",
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=False
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=df_validation,
    directory=os.getcwd()+"\\\\"+csv_handler.FROM_DATA_TO_PREPROCESSED_PATH,
    x_col=COLUMN_NAMES[-1],
    y_col=column_names_without_filename_and_id,
    target_size=(TRAIN_WIDTH, TRAIN_HEIGHT),
    color_mode="grayscale",
    class_mode="other",
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=False
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=os.getcwd()+"\\\\"+csv_handler.FROM_DATA_TO_PREPROCESSED_PATH,
    x_col=COLUMN_NAMES[-1],
    y_col=column_names_without_filename_and_id,
    target_size=(TRAIN_WIDTH, TRAIN_HEIGHT),
    color_mode="grayscale",
    class_mode="other",
    batch_size=1,
    shuffle=True,
    validate_filenames=False
)

# create the model

model = keras.Sequential([
    layers.Dropout(0.2, input_shape=(TRAIN_WIDTH, TRAIN_HEIGHT, 1)),
    layers.AveragePooling2D(2, 2),
    layers.Conv2D(16, 3, activation=HIDDEN_LAYER_ACTIVATION),
    layers.AveragePooling2D(2, 2),
    layers.Conv2D(32, 3, activation=HIDDEN_LAYER_ACTIVATION),
    layers.AveragePooling2D(2, 2),
    layers.Conv2D(64, 3, activation=HIDDEN_LAYER_ACTIVATION),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(512, activation=HIDDEN_LAYER_ACTIVATION),
    layers.Dropout(0.2),
    layers.Dense(OUTPUT_NEURON_NUMBER, activation=OUTPUT_LAYER_ACTIVATION)
])

# train on the model

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Compile the model")

model.compile(
    optimizer=keras.optimizers.Adam(0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[METRICS])

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Starting the search")

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode="min")]
)

print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Saving the best model")
