import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

face_cascade = cv.CascadeClassifier(
    r'C:\Users\Frank\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(
    r'C:\Users\Frank\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\cv2\data\haarcascade_eye.xml')

def face_filter(image, filter):
    width = image.shape[0]
    height = image.shape[1]

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    new_img = np.zeros((width, height, 4)).astype(np.uint8)

    for row in range(width):
        for column in range(height):
            new_img[row][column] = np.append(img[row][column], 255)

    if filter == 'batman':
        filter_img = cv.imread('batman.png', -1)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            height_mask = int(h * 1.5)
            width_mask = int(w * 0.87)
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + int(y / 25)][column + int(x * 1.08)] = filter_img[row][column]

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

    cv.imshow('img', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return

img = cv.imread('test.jpg')
face_filter(img, 'batman')
