import cv2
import debug_logging as debug


def count_points(dice):
    def get_x(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return x

    def get_x_w(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return x + w

    def get_y(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return y

    def get_y_h(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return y + h

    def is_contour_on_upper_half(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return y + h <= dice.shape[0] * 0.5

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
                # debug.showContour(dice_mask, cont)
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
    if len(selected) > 1:
        min_x = min(map(get_x, selected))
        max_x = max(map(get_x_w, selected))
        min_y = min(map(get_y, selected))
        max_y = max(map(get_y_h, selected))
        dice_mask = dice_mask[min_x - 3:max_x + 3, min_y - 3:max_y + 3]
        if dice_mask.shape[0] > dice_mask.shape[1]:
            selected = list(filter(is_contour_on_upper_half, selected))
    return len(selected)
