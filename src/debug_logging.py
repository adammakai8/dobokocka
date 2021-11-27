import cv2


def imshowAndPause(winname, image):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


def showContour(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    im2show = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(im2show, (x, y), (x + w, y + h), (0, 0, 255), 1)
    imshowAndPause('contour', im2show)
