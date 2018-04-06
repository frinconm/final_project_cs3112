import numpy as np
import cv2 as cv
import math as mt

face_cascade = cv.CascadeClassifier(
    r'C:\Users\Frank\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(
    r'C:\Users\Frank\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\cv2\data\haarcascade_eye.xml')


def get_matrix(image, theta, phi, gamma, dx, dy, dz):
    w = image.shape[1]
    h = image.shape[0]
    f = dz

    # Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 1],
                   [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                   [0, 1, 0, 0],
                   [np.sin(phi), 0, np.cos(phi), 0],
                   [0, 0, 0, 1]])

    RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                   [np.sin(gamma), np.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([[f, 0, w / 2, 0],
                   [0, f, h / 2, 0],
                   [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))


def rotation3d(image, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
    # width = image.shape[1]
    # height = image.shape[0]
    # shape = image.shape[2]
    # extra_img = np.zeros((height * 2, width * 2, shape)).astype(np.uint8)
    # extra_img[(int(height / 2)):(int(height / 2 + height)), int(width / 2): int(width / 2 + width)] = image
    # image = extra_img
    # Get radius of rotation along 3 axes
    rtheta = np.deg2rad(theta)
    rphi = np.deg2rad(phi)
    rgamma = np.deg2rad(gamma)

    # Get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal

    # Get projection matrix
    mat = get_matrix(image, rtheta, rphi, rgamma, dx, dy, dz)

    image = cv.warpPerspective(image, mat, (image.shape[1], image.shape[0]))
    # Crop

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)

    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv.boundingRect(cnt)

    crop = image[y:y + h, x:x + w]
    return crop


def face_filter(image, filter, crop_height, crop_width):
    height = image.shape[0]
    width = image.shape[1]

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    new_img = np.zeros((height, width, 4)).astype(np.uint8)

    for row in range(height):
        for column in range(width):
            new_img[row][column] = np.append(image[row][column], 255)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
    for (x, y, w, h) in faces:
        # cv.rectangle(new_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        # for (ex, ey, ew, eh) in eyes:
        #     cv.rectangle(new_img, (ex + x, ey + y), (ex + ew + x, ey + eh + y), (0, 255, 0), 2)

        if (len(eyes) == 2):
            # Calculating z angle
            eye1_center = (eyes[0][0] + eyes[0][2] / 2 + x, eyes[0][1] + eyes[0][3] / 2 + y)
            eye2_center = (eyes[1][0] + eyes[1][2] / 2 + x, eyes[1][1] + eyes[1][3] / 2 + y)
            face_z_angle = np.rad2deg(np.arctan((eye2_center[1] - eye1_center[1]) / (eye2_center[0] - eye1_center[0])))

            # Calculating x angle
            eyes_x_center = (eye1_center[0] + eye2_center[0]) / 2
            face_x_center = w / 2 + x
            x_off = eyes_x_center - face_x_center

            x_radius = w / 2
            z_off = mt.sqrt(x_radius ** 2 - x_off ** 2)
            face_x_angle = np.rad2deg(np.arctan(x_off / z_off))

            if face_x_angle > 35:
                face_x_angle = 35
            if face_x_angle < -35:
                face_x_angle = -35

            eyes_y_center = (eye1_center[1] + eye2_center[1]) / 2
            faces_data = np.append(faces_data,
                                   [[x, y, w, h, face_x_angle, 0, face_z_angle, eyes_x_center, eyes_y_center]], axis=0)
        else:
            print("Eye detection error")

    faces_data = np.delete(faces_data, 0, 0)

    if filter == 'anonymous':
        filter_img = cv.imread('filters/anonymous.png', -1)

        for face in faces_data:
            height_mask = int(face[3] * 1.06)
            width_mask = int(face[2] * 0.83)
            filter_img = rotation3d(filter_img, 0, face[4], face[6])
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            start_mask_x = int(face[7] - width_mask / 2.1)
            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + int(face[8] - height_mask / 2.9)][column + start_mask_x] = filter_img[row][column]

    elif filter == 'carnival':
        filter_img = cv.imread('filters/carnival.png', -1)

        for face in faces_data:
            height_mask = int(face[3] * 0.45)
            width_mask = int(face[2] * 0.83)
            filter_img = rotation3d(filter_img, 0, face[4], face[6])
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + int(face[8] - height_mask / 1.85)][column + int(face[7] - width_mask / 2.1)] = \
                        filter_img[row][column]

    elif filter == 'devil':
        filter_img = cv.imread('filters/devil.png', -1)

        for face in faces_data:
            height_mask = int(face[3] * 0.95)
            width_mask = int(face[2] * 1.24)
            filter_img = rotation3d(filter_img, 0, face[4], face[6])
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + int(face[8] - height_mask / 1.6)][column + int(face[7] - width_mask / 2.18)] = \
                            filter_img[row][column]

    elif filter == 'clown':
        filter_img = cv.imread('filters/clown.png', -1)

        for face in faces_data:
            height_mask = int(face[3] * 2.2)
            width_mask = int(face[2] * 2.3)
            filter_img = rotation3d(filter_img, 0, face[4], face[6])
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + int(face[8] - height_mask / 2)][column + int(face[7] - width_mask / 2)] = \
                            filter_img[row][column]

    elif filter == 'donald':
        filter_img = cv.imread('filters/donald.png', -1)

        for face in faces_data:
            height_mask = int(face[3] * 1.5)
            width_mask = int(face[2] * 1.1)
            filter_img = rotation3d(filter_img, 0, face[4], face[6])
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + int(face[8] - height_mask / 2)][column + int(face[7] - width_mask / 2)] = \
                            filter_img[row][column]

    elif filter == 'skull':
        filter_img = cv.imread('filters/skull.png', -1)

        for face in faces_data:
            height_mask = int(face[3] * 1.45)
            width_mask = int(face[2] * 1.05)
            filter_img = rotation3d(filter_img, 0, face[4], face[6])
            filter_img = cv.resize(filter_img, (width_mask, height_mask))

            for row in range(height_mask):
                for column in range(width_mask):
                    if filter_img[row][column][3] != 0:
                        new_img[row + int(face[8] - height_mask / 1.95)][column + int(face[7] - width_mask / 2.1)] = \
                            filter_img[row][column]

    new_img = new_img[(int(crop_height / 2)):(int(crop_height / 2 + crop_height)),
              int(crop_width / 2): int(crop_width / 2 + crop_width)]

    cv.imshow('img', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return


image_name = input("Type the name of your image: ")
img = cv.imread(image_name)
width = img.shape[1]
height = img.shape[0]
shape = img.shape[2]
extra_img = np.zeros((height * 2, width * 2, shape)).astype(np.uint8)
extra_img[(int(height / 2)):(int(height / 2 + height)), int(width / 2): int(width / 2 + width)] = img

print()
print("Select filter: ")
print("[1] V for Vendetta")
print("[2] Carnival Mask")
print("[3] Clown")
print("[4] Red Devil")
print("[5] Donald")
print("[6] Skull")
print()
option = int(input("> "))

if option == 1:
    face_filter(extra_img, 'anonymous', height, width)
elif option == 2:
    face_filter(extra_img, 'carnival', height, width)
elif option == 3:
    face_filter(extra_img, 'clown', height, width)
elif option == 4:
    face_filter(extra_img, 'devil', height, width)
elif option == 5:
    face_filter(extra_img, 'donald', height, width)
elif option == 6:
    face_filter(extra_img, 'skull', height, width)

print(option)