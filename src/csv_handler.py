import csv
import os
from constants import IMAGES_PATH
import numpy as np

CSV_FILE_NAME = "images.csv"
MIN_MAX_FILE_NAME = "min_max_values.txt"
COLUMN_NAMES = ["ID", "X11", "Y11", "X12", "Y12", "X21", "Y21", "X22", "Y22", "FILENAME"]
IMAGE_PATH = IMAGES_PATH.split("/")[-1]
FROM_PREPROCESSED_TO_DATA_PATH = "../data"
FROM_DATA_TO_PREPROCESSED_PATH = "../preprocessed"

min_max_array = []


def _normalize_image_names():
    images_names = os.listdir()
    images_array = []

    for image in images_names:
        columns = [int(item) for item in image.split(".")[0].split("_")]
        columns.pop()
        columns.pop()
        columns = np.array(columns)
        images_array.append(columns)

    images_array = np.array(images_array)
    images_array = images_array.transpose()

    normalized_result_array = []

    for array in images_array:
        min_p = min(array)
        max_p = max(array)

        min_max_array.append((min_p, max_p))

        normalized_array = array

        for element in array:
            value = (element - min_p) / (max_p - min_p)
            normalized_array = np.where(normalized_array == element, value, normalized_array)

        normalized_result_array.append(normalized_array)

    normalized_result_array = np.transpose(normalized_result_array)

    return normalized_result_array


def create_csv():
    _id = 0

    if not os.getcwd().endswith(IMAGE_PATH):
        os.chdir(IMAGES_PATH)

    os.chdir(FROM_PREPROCESSED_TO_DATA_PATH)

    if os.path.exists(CSV_FILE_NAME):
        os.remove(CSV_FILE_NAME)

    if os.path.exists(MIN_MAX_FILE_NAME):
        os.remove(MIN_MAX_FILE_NAME)

    os.chdir(FROM_DATA_TO_PREPROCESSED_PATH)

    image_names = os.listdir()
    normalized_vectors = _normalize_image_names()

    os.chdir(FROM_PREPROCESSED_TO_DATA_PATH)

    with open(CSV_FILE_NAME, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMN_NAMES)
        writer.writeheader()
        for image in normalized_vectors:
            row = {}
            index = 1
            for element in image:
                row[COLUMN_NAMES[index]] = element
                index += 1
            row[COLUMN_NAMES[index]] = image_names[_id]
            writer.writerow(row)
            _id += 1

    with open(MIN_MAX_FILE_NAME, "w") as file:
        for element in min_max_array:
            file.write(f"{element[0]} {element[1]}\n")
