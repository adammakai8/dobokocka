""" A kockák szegmentálását végző függvényeket tároló modul  """
import os

import cv2
import numpy as np
from preprocessing import process_image
from yolo_predict import predict_bounding_boxes
from segmentation import create_segment
from point_detection import count_points
from constants import TEMP_IMAGE_NAME


def segment_with_traditional_techniques(path):

    def cut_image_by_contour(img, contour_data):
        pt1 = (contour_data[1][0], contour_data[1][1])
        pt2 = (contour_data[1][0] + contour_data[1][2],
               contour_data[1][1] + contour_data[1][3])
        return img[pt1[1]: pt2[1], pt1[0]: pt2[0]]

    def morhp_filter(img):
        img = cv2.erode(img, np.ones((7, 7), np.uint8))
        img = cv2.dilate(img, np.ones((7, 7), np.uint8))
        img = cv2.dilate(img, np.ones((7, 7), np.uint8))
        img = cv2.erode(img, np.ones((7, 7), np.uint8))
        return img

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blockSize = 85
    C = -5
    if image.shape[0] < 900:
        blockSize = 35
        C = -5
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    img_mask = cv2.adaptiveThreshold(morhp_filter(image), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
    img_mask_filtered = morhp_filter(img_mask)
    contours, hierarchy = cv2.findContours(img_mask_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_with_info = [(cont, cv2.boundingRect(cont)) for cont in contours]
    contours_with_info.sort(key=lambda e: e[1][2] * e[1][3])
    contours_with_info.pop()
    pts1 = 0
    pts2 = 0
    while len(contours_with_info) > 0:
        pts1 = count_points(cut_image_by_contour(image, contours_with_info.pop()))
        if pts1 > 0:
            break
    while len(contours_with_info) > 0:
        pts2 = count_points(cut_image_by_contour(image, contours_with_info.pop()))
        if pts2 > 0:
            break

    return pts1 + pts2


def segment_with_cnn(path):
    image, scale = process_image(path)
    cv2.imwrite(TEMP_IMAGE_NAME, image)
    boxes = predict_bounding_boxes(TEMP_IMAGE_NAME)
    os.remove(TEMP_IMAGE_NAME)
    checked_boxes = 0
    pts = 0
    while checked_boxes < 2 and len(boxes) > 0:
        tmp = count_points(create_segment(boxes.pop(), image))
        if tmp > 0:
            pts += tmp
            checked_boxes += 1
    return pts
