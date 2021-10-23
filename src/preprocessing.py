""" Input képek előfeldolgozása, hogy könnyebben fel tudja dolgozni a neuronháló """

import cv2
from os import listdir

imgs = '../res/images/'
resdir = '../res/preprocessed/'

dest_shape = (450, 600)


def process_image(path):
    """
    Egy képet fájlnév alapján megkeres és feldolgoz
    :param path: a kép fájlneve elérési út nélkül
    :return: a feldolgozott kép
    """
    label_data = path.split('.')[0].split('_')
    label_data_int = [int(item) for item in label_data]
    image = cv2.imread(f'{imgs}{path}')
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
    if shape[0] > dest_shape[0] and shape[1] > dest_shape[1]:
        scale = shape[0] / dest_shape[0]
        image = cv2.resize(image, dest_shape)
        label_data_int = [item * scale for item in label_data_int]
    cv2.imwrite(f'{resdir}{"_".join(str(item) for item in label_data_int)}.png', image)
    return image


def process_all():
    for picture in listdir(imgs):
        process_image(picture)

# Ha az összes feldolgozatlan képet szeretnénk előfeldolgozni, akkor kommenteljük ki az alsó sort
# process_all()
