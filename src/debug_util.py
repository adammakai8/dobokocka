import cv2
import numpy as np
import point_detection as pd
import tensorflow.keras as ks
import segmentation as sg
import detection_segmentation as ds
import debug_logging as debug
import os


def test_point_detection():
    IMAGE_NAME = '133_354_167_388_388_66_411_80_0'

    dice1 = cv2.imread(f'../res/segmented/{IMAGE_NAME}/{IMAGE_NAME}_1.png', cv2.IMREAD_GRAYSCALE)
    dice2 = cv2.imread(f'../res/segmented/{IMAGE_NAME}/{IMAGE_NAME}_2.png', cv2.IMREAD_GRAYSCALE)

    # pd.point_detection(dice1, dice2)


def test_neuron_model():
    MODEL_PATH = '../res/data/model.hdf5'
    MIN_MAX_PATH = '../res/data/min_max_values.txt'
    TEST_IMAGE_PATH = '../res/preprocessed/108_82_141_135_340_326_415_407_0.png'

    min_max_values = []
    with open(MIN_MAX_PATH, 'r') as file:
        while True:
            linestr = file.readline()
            if linestr == '':
                break
            min_max_values.append([int(linestr.split(' ')[0]), int(linestr.split(' ')[1])])

    model = ks.models.load_model(MODEL_PATH, compile=False)
    image = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    image = image / 255
    image = np.expand_dims(image, (0, 3))

    result = model.predict(image)[0]
    result_denorm = []
    for i in range(len(result)):
        result_denorm.append(int(round(result[i] * (min_max_values[i][1] - min_max_values[i][0]) + min_max_values[i][0])))

    print(result_denorm)
    for i in range(0, len(result_denorm), 2):
        temp = result_denorm[i]
        result_denorm[i] = result_denorm[i + 1]
        result_denorm[i + 1] = temp


def trad_segment_statistics():
    IMAGES_PATH = '../res/images/'
    labels = []
    values = []
    for path in os.listdir(IMAGES_PATH):
        if path.split('.')[1] != 'jpg' and path.split('.')[1] != 'png':
            continue
        # img = cv2.imread(f'{IMAGES_PATH}{path}', cv2.IMREAD_GRAYSCALE)
        # if img.shape[0] <= 900:
        #     continue
        labels.append(int(path.split('.')[0].split('_')[-1]))
        values.append(ds.segment_with_traditional_techniques(f'{IMAGES_PATH}{path}'))
    differences = []
    for i in range(len(labels)):
        differences.append(abs(labels[i] - values[i]))

    print('Statistics:')
    print(labels)
    print(values)
    print(differences)
    print('-----------------------------------------------------------------------')
    print(f'correct guesses: {len(list(filter(lambda x: x == 0, differences)))}/{len(labels)}')
    print(f'max error: {max(differences)}')
    print(f'min error: {min(differences)}')
    print(f'avg error: {sum(differences) / len(differences)}')
    print(f'median error: {np.median(differences)}')
    print('-----------------------------------------------------------------------')
    for i in range(12):
        print(f'error {i}: {len(list(filter(lambda x: x == i, differences)))}')


''' Innentől lehet függvényeket meghívni, de ha más szkriptben használod a debug_util-t, akkor előtte kommentezd
 ki ezeket a hívásokat '''
# test_neuron_model()
# test_point_detection()
trad_segment_statistics()
