import os
import copy
from datetime import datetime

import pandas as pd
from matplotlib import pyplot
from matplotlib import image

from tensorflow import keras
from tensorflow.keras import layers
from keras_preprocessing.image import ImageDataGenerator

import csv_handler

from constants import IMAGES_PATH, TRAIN_WIDTH, TRAIN_HEIGHT, HIDDEN_LAYER_ACTIVATION, \
    OUTPUT_NEURON_NUMBER, OUTPUT_LAYER_ACTIVATION, METRICS, EPOCHS, BATCH_SIZE
from csv_handler import COLUMN_NAMES

# const variables

MODEL_FILE = "model.txt"

_TRAIN_PCT = 0.8
_VALIDATION_PCT = 0.1

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

dropout1 = [0.1, 0.2, 0.3]
dropout2 = [0.3, 0.4, 0.5]
dropout3 = [0.2, 0.3, 0.4]
dropout4 = [0.2, 0.3, 0.4]

conv2d1 = [8, 16, 32, 64]
conv2d2 = [16, 32, 64]
conv2d3 = [32, 64]

dense1 = [128, 256, 512, 1024]
dense2 = [256, 512, 1024]

learning_rate = [0.01, 0.007, 0.004, 0.002, 0.001, 0.0007, 0.0004, 0.0002, 0.0001]

model_dict = {}
best_model = 0
best_val_accuracy = 0

try:
    for de1 in range(4):
        for cv2d1 in range(4):
            for cv2d3 in range(2):
                for cv2d2 in range(3):
                    for cv2dn in range(3):
                        for d2 in range(3):
                            for de2 in range(3):
                                for d4 in range(3):
                                    for de in range(2):
                                        for d1 in range(3):
                                            for d3 in range(3):
                                                for lr in range(9):
                                                    model = keras.Sequential([
                                                        layers.Dropout(dropout1[d1],
                                                                       input_shape=(TRAIN_WIDTH, TRAIN_HEIGHT, 1)),
                                                        layers.AveragePooling2D(2, 2),
                                                        layers.Conv2D(conv2d1[cv2d1], 3,
                                                                      activation=HIDDEN_LAYER_ACTIVATION)
                                                    ])
                                                    if cv2dn >= 1:
                                                        model.add(layers.AveragePooling2D(2, 2))
                                                        model.add(layers.Conv2D(conv2d2[cv2d2], 3,
                                                                                activation=HIDDEN_LAYER_ACTIVATION))
                                                    if cv2dn >= 2:
                                                        model.add(layers.AveragePooling2D(2, 2))
                                                        model.add(layers.Conv2D(conv2d3[cv2d3], 3,
                                                                                activation=HIDDEN_LAYER_ACTIVATION))
                                                    model.add(layers.Dropout(dropout2[d2]))
                                                    model.add(layers.Flatten())
                                                    model.add(
                                                        layers.Dense(dense1[de1], activation=HIDDEN_LAYER_ACTIVATION))
                                                    model.add(layers.Dropout(dropout3[d3]))
                                                    if de >= 1:
                                                        model.add(layers.Dense(dense2[de2],
                                                                               activation=HIDDEN_LAYER_ACTIVATION))
                                                        model.add(layers.Dropout(dropout3[d4]))
                                                    model.add(layers.Dense(OUTPUT_NEURON_NUMBER,
                                                                           activation=OUTPUT_LAYER_ACTIVATION))

                                                    # train on the model

                                                    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[
                                                          :-1] + f": Compiling model {de1}_{cv2d1}_{cv2d3}_{cv2d2}_{cv2dn}_{d2}_{de2}_{de}_{d1}_{d3}_{d4}_{lr}")

                                                    model.compile(
                                                        optimizer=keras.optimizers.Adam(learning_rate[lr]),
                                                        loss=keras.losses.MeanSquaredError(),
                                                        metrics=[METRICS])

                                                    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[
                                                          :-1] + ": Starting the learning")

                                                    history = model.fit(
                                                        train_generator,
                                                        epochs=EPOCHS,
                                                        validation_data=validation_generator,
                                                        callbacks=[
                                                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                                                                          mode="min", restore_best_weights=True)]
                                                    )

                                                    if best_model == 0 or min(
                                                            history.history['val_loss']) < best_val_accuracy:
                                                        print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[
                                                              :-1] + ": Current modell is better, so save it")
                                                        model_dict.clear()
                                                        model_dict['de'] = de
                                                        model_dict['de1'] = dense1[de1]
                                                        model_dict['de2'] = dense2[de2]
                                                        model_dict['cv2dn'] = cv2dn
                                                        model_dict['cv2d1'] = conv2d1[cv2d1]
                                                        model_dict['cv2d2'] = conv2d2[cv2d2]
                                                        model_dict['cv2d3'] = conv2d3[cv2d3]
                                                        model_dict['d1'] = dropout1[d1]
                                                        model_dict['d2'] = dropout2[d2]
                                                        model_dict['d3'] = dropout3[d3]
                                                        model_dict['d4'] = dropout4[d4]
                                                        model_dict['lr'] = lr
                                                        best_model = model
                                                        best_val_accuracy = min(history.history['val_loss'])

                                                        best_model.save('./model.hdf5')
                                                        with open(MODEL_FILE, "w") as file:
                                                            for key in model_dict.keys():
                                                                file.write(f"{key}:{model_dict[key]}\n")
except:
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": An exception has been throwed")
finally:
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f:')[:-1] + ": Write the best model into a file")

    best_model.save('./model.hdf5')
    with open(MODEL_FILE, "w") as file:
        for key in model_dict.keys():
            file.write(f"{key}:{model_dict[key]}\n")
