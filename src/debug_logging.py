import cv2


def imshowAndPause(winname, image):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
