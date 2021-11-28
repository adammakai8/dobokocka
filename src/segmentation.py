""" Szegmentálást és a szegmensek áramlását megvalósító függvények """

import cv2
from os import listdir
from os import mkdir

import debug_logging as debug

prep_dir = '../res/preprocessed/'
seg_dir = '../res/segmented/'


def create_and_save_segments_from_preprocessed():
    """
    Az előfeldolgozott képek címkéi alapján kivágja a dobókockákat és elmenti a res/segmented mappába.
    Tesztelési célokra használatos.
    """
    for pic in listdir(prep_dir):
        dirname = pic.split(".")[0]
        try:
            mkdir(f'{seg_dir}{dirname}')
        except FileExistsError:
            continue
        image = cv2.imread(f'{prep_dir}{pic}', cv2.IMREAD_GRAYSCALE)
        labels = [int(item) for item in pic.split('.')[0].split('_')]
        dice1 = image[labels[1]:labels[3], labels[0]:labels[2]]
        dice2 = image[labels[5]:labels[7], labels[4]:labels[6]]
        cv2.imwrite(f'{seg_dir}{dirname}/{pic.split(".")[0]}_1.png', dice1)
        cv2.imwrite(f'{seg_dir}{dirname}/{pic.split(".")[0]}_2.png', dice2)


def create_segment(coordinates, image):
    dice = image[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
    return dice


# Ha külön akartok szegmenseket kimenteni tesztadatként az alsó sor kommentezzétek ki
# create_and_save_segments_from_preprocessed()
