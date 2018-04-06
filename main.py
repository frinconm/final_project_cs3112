import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

face_cascade = cv.CascadeClassifier(
    r'C:\Users\frank\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(
    r'C:\Users\frank\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_eye.xml')


def face_filter(image, filter, crop_height, crop_width):
    height = image.shape[0]
    width = image.shape[1]

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    new_img = np.zeros((height, width, 4)).astype(np.uint8)

    for row in range(height):
        for column in range(width):
            new_img[row][column] = np.append(image[row][column], 255)

    if filter == 'batman':
        filter_img = cv.imread('batman.png', -1)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # roi_gray = gray[y:y + h, x:x + w]
            # roi_color = image[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh) in eyes:
            #     cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            height_mask = int(h * 1.5)
            width_mask = int(w * 0.87)
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + y - int(height_mask / 2.6)][column + x] = filter_img[row][column]

    elif filter == 'carnival':
        filter_img = cv.imread('carnival.png', -1)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            height_mask = int(h * 0.5)
            width_mask = int(w)
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + int(y * 1.2)][column + int(x * 1.02)] = filter_img[row][column]

    new_img = new_img[(int(crop_height / 2)):(int(crop_height / 2 + crop_height)),
              int(crop_width / 2): int(crop_width / 2 + crop_width)]

    cv.imshow('img', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return


img = cv.imread('test.jpg')
width = img.shape[1]
height = img.shape[0]
shape = img.shape[2]
extra_img = np.zeros((height * 2, width * 2, shape)).astype(np.uint8)
extra_img[(int(height / 2)):(int(height / 2 + height)), int(width / 2): int(width / 2 + width)] = img
face_filter(extra_img, 'batman', height, width)
