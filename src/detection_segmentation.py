""" Under construction! Testing various techniques  """

import cv2
import numpy as np


def cut_image(img):
    img_blurred = cv2.GaussianBlur(img, (7, 7), 4.0)
    img_edges = cv2.Canny(img_blurred, 140, 250, None, 5, True)
    result = np.where(img_edges == 255)
    return img[min(result[0]): max(result[0]), min(result[1]): max(result[1])]


def contour(img):

    def has_dots(candidate):
        # TODO: ha túl kicsi a képrészlet, nagyítani kell hogy ne legyenek a pöttyök rajta kockásak
        mask = cv2.adaptiveThreshold(candidate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, -9)
        cv2.imshow(f'masked{candidate.shape[0]}-{candidate.shape[1]}', mask)
        # cv2.waitKey(0)
        im_ar = candidate.shape[0] * candidate.shape[1]
        dot_candidates, h = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for dc in dot_candidates:
            ar = cv2.contourArea(dc)
            if ar < im_ar / 1000 or ar > im_ar * 0.4:
                continue
            ker = cv2.arcLength(dc, True)
            if abs(ar / ker**2 - 0.08) < 0.01:
                return True
        return False

    img_mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -15)
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img_area = img.shape[0] * img.shape[1]
    cv2.imshow('mask', img_mask)
    for cont in contours:
        x, y, width, height = cv2.boundingRect(cont)
        area = width * height
        if area < img_area / 1000 or area > img_area * 0.9 or width / height < 1 / 5 or width / height > 5:
            continue
        dice = img[y: y + height, x: x + width]
        if not has_dots(np.copy(dice)):
            continue
        cv2.imshow(f'segmented{x}-{y}', dice)


def edges(img):
    img_blurred = cv2.GaussianBlur(img, (7, 7), 4.0)
    # cv2.imshow('blurred', img_blurred)
    img_edges = cv2.Canny(img_blurred, 140, 250, None, 5, True)
    cv2.imshow('edges', img_edges)


imgs = '../res/images'
n = 3

for i in range(n, n + 1):
    image = cv2.imread(f'{imgs}/Dice{i}.png', cv2.IMREAD_GRAYSCALE)
    # image = cut_image(image)
    cv2.imshow('input', image)
    contour(image)
    # edges(image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
