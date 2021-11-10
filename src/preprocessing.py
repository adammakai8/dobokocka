""" Input képek előfeldolgozása, hogy könnyebben fel tudja dolgozni a neuronháló """

import cv2
from os import listdir

IMG_PATH = '../res/images/'
RESDIR_PATH = '../res/preprocessed/'

DEST_SHAPE = (450, 600)


def process_image(path):
    """
    Egy képet fájlnév alapján megkeres és feldolgoz
    :param path: a kép fájlneve elérési út nélkül
    :return: a feldolgozott kép
    """
    if path.count('/') > 0 or path.count('\\') > 0:
        path.replace('\\', '/')
        image_name = path.split('/')[-1]
    else:
        image_name = path
        path = f'{IMG_PATH}{path}'
    label_data = image_name.split('.')[0].split('_')
    label_data_int = [int(item) for item in label_data]
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = image.shape
    if shape[0] > shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        shape = image.shape
        for i in range(0, 5, 4):
            x1 = label_data_int[i]
            label_data_int[i] = shape[1] - label_data_int[i + 3]
            y1 = label_data_int[i + 1]
            label_data_int[i + 1] = x1
            x2 = label_data_int[i + 2]
            label_data_int[i + 2] = shape[1] - y1
            label_data_int[i + 3] = x2
    if shape[0] / shape[1] != 0.75:
        return -1
    if shape[0] > DEST_SHAPE[0] and shape[1] > DEST_SHAPE[1]:
        scale = DEST_SHAPE[0] / shape[0]
        image = cv2.resize(image, (DEST_SHAPE[1], DEST_SHAPE[0]))
        label_data_int = [int(item * scale) for item in label_data_int]
    return image, label_data_int


def process_all():
    for path in listdir(IMG_PATH):
        image, label_data_int = process_image(path)
        cv2.imwrite(f'{RESDIR_PATH}{"_".join(str(item) for item in label_data_int)}.png', image)


# Ha az összes feldolgozatlan képet szeretnénk előfeldolgozni, akkor kommenteljük ki az alsó sort
# process_all()
