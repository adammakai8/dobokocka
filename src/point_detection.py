""" A kivágott kockákat tartalmazó részképletek feldolgozását, pontok számolását végző modul """
import cv2


def count_points(dice):

    def is_overlapping(contour, prev_contours):
        x, y, w, h = cv2.boundingRect(contour)
        for prev in prev_contours:
            x_p, y_p, w_p, h_p = cv2.boundingRect(prev)
            if ((x_p < x < x_p + w_p or x_p < x + w < x_p + w_p) and (y_p < y < y_p + h_p or y_p < y + h < y_p + h_p)) \
                    or ((x < x_p < x + w or x < x_p + w_p < x + w) and (y < y_p < y + h or y < y_p + h_p < y + h)):
                return True
        return False

    def examine_contours(contour_list):
        img_area = dice_mask.shape[0] * dice_mask.shape[1]
        min_area_ratio = 100 if img_area > 30000 else 250
        selected_contours = []
        for cont in contour_list:
            if img_area / min_area_ratio < cv2.contourArea(cont) < img_area / 3 \
                    and cv2.contourArea(cont) / cv2.arcLength(cont, True) ** 2 > 0.055 \
                    and not is_overlapping(cont, selected_contours):
                selected_contours.append(cont)
        return selected_contours

    if dice.shape[0] * dice.shape[1] <= 1000:
        return 0
    blockSize = 35
    C = -15
    if dice.shape[0] * dice.shape[1] <= 30000:
        blockSize = 55
        C = -15
    dice_mask = cv2.adaptiveThreshold(
        cv2.medianBlur(dice, 3), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
    contours, hierarchy = cv2.findContours(dice_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    selected = examine_contours(contours)
    return len(selected)
