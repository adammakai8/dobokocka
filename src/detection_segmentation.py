""" Under construction! Testing various techniques  """

import cv2
import numpy as np
import preprocessing as prep

import point_detection as pd

# import tensorflow.keras as ks


DATA_PATH = '../res/data'


def segment_with_traditional_techniques(path):

    def cut_image_by_contour(img, contour_data):
        pt1 = (contour_data[1][0], contour_data[1][1])
        pt2 = (contour_data[1][0] + contour_data[1][2],
               contour_data[1][1] + contour_data[1][3])
        return img[pt1[1]: pt2[1], pt1[0]: pt2[0]]

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # 55 -15 (25, -15)
    blockSize = 75
    C = -5
    if image.shape[0] < 900:
        blockSize = 35
        C = -5
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
        # cv2.imshow('image', image)
    img_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
    img_mask_dilated = cv2.dilate(img_mask, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(img_mask_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    size = (int(img_mask_dilated.shape[1] / 4), int(img_mask_dilated.shape[0] / 4))
    # cv2.imshow('mask', cv2.resize(img_mask_dilated, size))
    contours_with_info = [(cont, cv2.boundingRect(cont)) for cont in contours]
    contours_with_info.sort(key=lambda e: e[1][2] * e[1][3])
    contours_with_info.pop()
    pts1 = 0
    pts2 = 0
    while len(contours_with_info) > 0:
        pts1 = pd.count_points(cut_image_by_contour(image, contours_with_info.pop()))
        if pts1 > 0:
            break
    while len(contours_with_info) > 0:
        pts2 = pd.count_points(cut_image_by_contour(image, contours_with_info.pop()))
        if pts2 > 0:
            break

    return pts1 + pts2


def segment_with_cnn(path):
    image, labels = prep.process_image(path)
    # model = ks.models.load_model(f'{DATA_PATH}/model.hdf5', compile=False)
    # print(model.predict(image))
