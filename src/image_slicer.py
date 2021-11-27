import cv2
import numpy as np
import os

from constants import IMAGES_PATH

os.chdir(IMAGES_PATH)

SIZE = 270000
INTERVAL_MIN, INTERVAL_MAX = 200, 255

line_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255)]

for image in os.listdir():
    gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    gray = cv2.medianBlur(gray, 9)
    img_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, -20)
    dst = cv2.dilate(img_mask, np.ones((5, 5), np.uint8))
    dst = cv2.erode(dst, np.ones((5, 5), np.uint8))
    dst = cv2.erode(dst, np.ones((5, 5), np.uint8))
    dst = cv2.dilate(dst, np.ones((5, 5), np.uint8))

    cv2.imshow("segmented", dst)

    white_pixel = np.count_nonzero(dst)

    # percentage = 0.0
    # right_index = 1
    #
    # image_left = 0
    # foundTheDice = False
    #
    # while True:
    #     image_left = dst[:, 0:right_index]
    #     white_pixel_left_percentage = np.count_nonzero(image_left) / white_pixel
    #     if foundTheDice or 0.48 <= white_pixel_left_percentage <= 0.52:
    #         foundTheDice = True
    #         image_slice = dst[:, right_index:right_index + 1]
    #         if np.count_nonzero(image_slice) > 0:
    #             right_index += 1
    #             continue
    #         else:
    #             break
    #     right_index += 1
    #
    # if np.count_nonzero(image_left) / white_pixel >= 0.8:
    #     while True:
    #         image_slice = dst[:, right_index - 1:right_index]
    #         if np.count_nonzero(image_slice) > 0:
    #             right_index -= 1
    #             continue
    #         else:
    #             break

    # image_right = 0
    # left_index = dst.shape[1]
    # foundTheDice = False
    #
    # while True:
    #     image_right = dst[:, left_index:dst.shape[1]]
    #     white_pixel_left_percentage = np.count_nonzero(image_right) / white_pixel
    #     if foundTheDice or 0.7 <= white_pixel_left_percentage <= 1:
    #         foundTheDice = True
    #         image_slice = dst[:, left_index-1:left_index]
    #         if np.count_nonzero(image_slice) > 0:
    #             left_index -= 1
    #             continue
    #         else:
    #             break
    #     left_index -= 1
    #
    # # cv2.imshow("img_right", image_right)
    #
    # if np.count_nonzero(image_right) / white_pixel >= 0.8:
    #     while True:
    #         image_slice = dst[:, left_index:left_index+1]
    #         if np.count_nonzero(image_slice) > 0:
    #             left_index += 1
    #             continue
    #         else:
    #             break
    #
    # slicer = int(left_index + right_index / 2)

    # cv2.imshow("left", dst[:, :right_index])
    # cv2.imshow("right", dst[:, right_index:])

    cv2.waitKey(0)
