#!/usr/bin/env python3

"""
    This is the main program for the iFaceDQ artwork shown in the Art Machines exhibition 2020.
    The program is developed from the original OpenCV C++ tutorial (https://docs.opencv.org) -
    Face swapping using face landmark detection. The python version also referenced the Satya
    Mallick's implementation in the Learn OpenCV website (https://www.learnopencv.com).

    There are two major functions of the program.

    matchFace
    The match face function will match the current face detected from the webcam, against a database
    of 75 Legislative Council members using the NearestNeighbors search from scikit-learn. The facial
    representation is handled by the 128 elements vector from the dlib package.

    swapFace
    After matching the visitor's face with a Council member, the swap face function will swap the member's
    face with the visitor's own. The actual swapping is performed by the Delaunay triangulation of the 68
    facial landmarks and then warping the triangles from the member's face to the visitor's.

"""

import sys
import numpy as np
import pandas as pd
import cv2
import dlib
import csv
import os
import math
import pygame
import matplotlib
import matplotlib.backends.backend_agg as agg
import pylab
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
matplotlib.use("Agg")


def findLandmark(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    max_area = sys.float_info.min
    max_face = None
    for face in faces:
        width = face.right() - face.left()
        height = face.bottom() - face.top()
        area = width * height
        if area > max_area:
            max_area = area
            max_face = face
    landmarks = None
    points = []
    if max_face is not None:
        landmarks = predictor(gray, max_face)
        for i in range(0, 68):
            points.append((landmarks.part(i).x, landmarks.part(i).y))
    return points


def readCsv(file):
    with open(file, 'rt') as csvFile:
        data = list(csv.reader(csvFile))

        face_slice = [data[i][1:] for i in range(0, len(data))]
        face_label = [data[i][0] for i in range(0, len(data))]
        face_array = np.array(face_slice, dtype=np.float)
        nbrs = NearestNeighbors(n_neighbors=1).fit(face_array)
    return face_label, nbrs


def readMembers(file):
    status = pd.read_csv(file, header=None).set_index(0).squeeze().to_dict()
    return status


def photoLandmarks(labels):
    photos = []
    landmarks = []
    for label in labels:
        img_file = os.path.join(photo_dir, label + '.jpg')
        img = cv2.imread(img_file)
        print('Reading facial landmarks ', label)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(grey)
        max_area = sys.float_info.min
        max_face = None
        for face in faces:
            width = face.right() - face.left()
            height = face.bottom() - face.top()
            area = width * height
            if area > max_area:
                max_area = area
                max_face = face

        landmark = None
        points = []
        if max_face is not None:
            landmark = predictor(grey, max_face)
            for i in range(0, 68):
                points.append((landmark.part(i).x, landmark.part(i).y))

        landmarks.append(points)
        photos.append(img)
    return photos, landmarks


def videoLandmarks(img):
    factor = 1.5
    points = []
    small = cv2.resize(img, (round(img.shape[1]/factor), round(img.shape[0]/factor)))
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = detector(grey)
    max_area = sys.float_info.min
    max_face = None
    for face in faces:
        width = face.right() - face.left()
        height = face.bottom() - face.top()
        area = width * height
        if area > max_area:
            max_area = area
            max_face = face

    landmarks = None
    if max_face is not None:
        landmarks = predictor(grey, max_face)
        for i in range(0, 68):
            points.append((round(landmarks.part(i).x*factor), round(landmarks.part(i).y*factor)))
    return points


def matchFace(face, shape):
    face_chip = dlib.get_face_chip(face, shape)
    descriptor = facerec.compute_face_descriptor(face_chip)
    desc = np.array([descriptor])
    distances, indices = nbrs.kneighbors(desc)
    distance = distances[0][0]
    idx = indices[0][0]
    name = face_label[idx]
    return idx, distance, name


def swapFace(img1, img2, idx):
    points1 = photo_landmarks[idx]
    points2 = videoLandmarks(img2)

    if len(points1) == 0 or len(points2) == 0:
        return img2
    triangles = findTriangle(img1, points1)
    img1x = np.float32(img1)
    img2x = np.float32(img2)
    warp = np.copy(img2x)

    t1 = findTrianglePoints(triangles, points1)
    t2 = findTrianglePoints(triangles, points2)

    for i in range(0, len(triangles)):
        warpTriangle(img1x, warp, t1[i], t2[i])

    hull2 = []
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull2.append(points2[int(hullIndex[i])])

    hull8U = []

    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), WHITE)
    r = cv2.boundingRect(np.float32([hull2]))
    if r[0] + r[2] > img2.shape[1] or r[1] + r[3] > img2.shape[0]:
        return img2
    if r[0] < 0 or r[1] < 0:
        return img2
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
    output = cv2.seamlessClone(np.uint8(warp), img2, mask, center, cv2.NORMAL_CLONE)
    return output


def warpTriangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    if r2[0] + r2[2] > img2.shape[1] or r2[1] + r2[3] > img2.shape[0]:
        return
    if r2[0] < 0 or r2[1] < 0:
        return

    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    warp_mat = cv2.getAffineTransform(np.float32(t1Rect), np.float32(t2Rect))
    img2Rect = cv2.warpAffine(img1Rect, warp_mat, size, None, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    img2Rect = img2Rect * mask

    submat = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = submat + img2Rect

    return


def findTrianglePoints(triangles, points):
    result = []
    for t in triangles:
        pts = []
        for p in t:
            pts.append(points[p])
        result.append(pts)
    return result


def findTriangle(img, points):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList()
    triangles = []
    for t in triangleList:
        pt = []
        pp = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        if contains(rect, pt[0]) and contains(rect, pt[1]) and contains(rect, pt[2]):
            for p in pt:
                idx = matchPoint(p, points)
                if idx != -1:
                    pp.append(idx)
            if len(pp) == 3:
                triangles.append(pp)
    return triangles


def matchPoint(pt, points):
    min_dist = 1.0
    result = -1
    for idx in range(len(points)):
        distance = math.dist(pt, points[idx])
        if distance < min_dist:
            result = idx
    return result


def contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# initialise all files

data_dir = os.getcwd() + os.path.sep + 'data'
photo_dir = os.getcwd() + os.path.sep + 'data' + os.path.sep + 'photos'
member_file = os.path.join(data_dir, 'LegislativeCouncilMembers.csv')
csv_file = os.path.join(data_dir, 'face_descriptor.csv')

predictor_path = os.path.join(data_dir, 'shape_predictor_5_face_landmarks.dat')
shape_file = os.path.join(data_dir, 'shape_predictor_68_face_landmarks.dat')
model_path = os.path.join(data_dir, 'dlib_face_recognition_resnet_model_v1.dat')

# initialise all detectors

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
predictor = dlib.shape_predictor(shape_file)
facerec = dlib.face_recognition_model_v1(model_path)

# initialise council members details

face_label, nbrs = readCsv(csv_file)
dq_status = readMembers(member_file)
portraits, photo_landmarks = photoLandmarks(face_label)
print(len(portraits))

# setup pygame and data

pygame.init()
pygame.font.init()
font1 = pygame.font.Font(os.path.join(data_dir, "NotoSansHK-Regular.otf"), 14)
font2 = pygame.font.Font(os.path.join(data_dir, "SourceCodePro-Regular.ttf"), 14)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
WORK_HOUR = (9, 19)
MAX_MSG = 15
dimen1 = [1280, 720]
dimen2 = [1280, 720]
dimen3 = [400, 400]
offset1 = ((dimen2[1] - dimen3[0]) // 2, dimen2[1] - dimen3[1])
offset2 = ((dimen2[0] - dimen2[1]) // 2, 0)
ratio = (dimen2[1] / dimen3[0], dimen2[1] / dimen3[1])

# definition of the pie chart

labels = ['Low Risk', 'High Risk', 'DQ']
colours = {'Low Risk': 'C0',
           'High Risk': 'C1',
           'DQ': 'C2'}
tally = [0, 0, 0]
explode = (0, 0, 0.1)
fig = pylab.figure(1, figsize=(3.6, 3.6), edgecolor='none', facecolor='none')
canvas = agg.FigureCanvasAgg(fig)
size = canvas.get_width_height()

fps = 30
factor = 2
name = ""
pname = ""
status = ""
photo_idx = -1
distance = 0
messages = []
running = True
clock = pygame.time.Clock()

# screen = pygame.display.set_mode(dimen1, pygame.HWSURFACE)
screen = pygame.display.set_mode(dimen1, pygame.FULLSCREEN | pygame.HWSURFACE)
pygame.display.set_caption('iFaceDQ')
imgSurf = pygame.Surface(dimen3)
buffer = pygame.Surface((dimen2[1], dimen2[1]))
panel = pygame.Surface((dimen1[1], dimen1[0]-dimen1[1]))
buffer.fill(BLACK)
panel.fill(BLACK)

# initialise OpenCV camera capture

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimen2[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimen2[1])
cap.set(cv2.CAP_PROP_FPS, fps)

pygame.mouse.set_visible(False)

now = datetime.now()
if now.hour < WORK_HOUR[0] or now.hour > WORK_HOUR[1]:
    running = False

while running:
    ret, frame = cap.read()
    if frame is None:
        continue

    now = datetime.now()
    screen.fill(BLACK)
    crop = frame[0:dimen2[1], offset2[0]:offset2[0] + dimen2[1]]
    crop = crop[offset1[1]:offset1[1] + dimen3[1], offset1[0]:offset1[0] + dimen3[0]]
    flip = cv2.flip(crop, 1)
    small = cv2.resize(flip, (flip.shape[1] // factor, flip.shape[0] // factor))
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dets = detector(grey, 1)
    face = None
    rect = None
    shape = None
    max_value = sys.float_info.min
    for k, d in enumerate(dets):
        area = abs(d.right() - d.left()) * abs(d.bottom() - d.top())
        if area > max_value:
            max_value = area
            face = d
            rect = (d.left() * factor, d.top() * factor,
                    d.right() * factor, d.bottom() * factor)

    if face is not None:
        shape = sp(small, face)

        photo_idx, distance, name = matchFace(small, shape)
        status = dq_status[name]
        tmp_img = portraits[photo_idx]

        if name != pname:
            pname = name
            for i in range(len(labels)):
                if status == labels[i]:
                    tally[i] = tally[i] + 1
            ts = now.strftime("%H:%M:%S")
            msg = now.strftime("%H:%M:%S") + " " + name.replace('_', ' ') + " (" + status + ")"
            if len(messages) < MAX_MSG:
                messages.append(msg)
            else:
                messages.pop(0)
                messages.append(msg)

        output = swapFace(tmp_img, flip, photo_idx)
        frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        frame = cv2.transpose(frame)
        frame = cv2.resize(frame, (dimen2[1], dimen2[1]), cv2.INTER_AREA)
        view = pygame.pixelcopy.make_surface(frame)
        buffer.blit(view, (0, 0))
        pygame.draw.rect(buffer, RED,
                         [rect[0]*ratio[0], rect[1]*ratio[1],
                          (rect[2] - rect[0])*ratio[0], (rect[3] - rect[1])*ratio[1]], 1)

        pygame.draw.rect(buffer, RED,
                         [rect[0]*ratio[0], rect[1]*ratio[1], (rect[2]-rect[0])*ratio[0], 20*ratio[1]], 0)
        pygame.draw.rect(buffer, RED,
                         [rect[0]*ratio[0], rect[3]*ratio[1], (rect[2]-rect[0])*ratio[1], 20*ratio[1]], 0)
        text1 = font1.render(name.replace('_', ' '), True, WHITE)
        text2 = font1.render("{:.3f}".format(distance), True, WHITE)
        text3 = font1.render(status, True, WHITE)
        buffer.blit(text1, (rect[0]*ratio[0]+10, rect[1]*ratio[1]+6))
        buffer.blit(text3, (rect[0]*ratio[0]+10, rect[3]*ratio[1]+6))
        screen.blit(pygame.transform.rotate(buffer, 90), (0, 0))
    else:
        name = ""
        status = ""

        buffer.fill(BLACK)
        screen.blit(buffer, (0, 0))

    if sum(tally) > 0:
        panel.fill(BLACK)
        pylab.clf()
        ax = fig.gca()
        ax.pie(tally, explode=explode, labels=labels, startangle=90, autopct='%1.0f%%',
               textprops={'color': "w"}, normalize=True,
               colors=[colours[key] for key in labels])

        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        surf = pygame.image.frombuffer(raw_data, size, 'RGBA')
        ypos = 100
        panel.blit(surf, [310, ypos])
        ypos = 150
        for i in range(len(messages)):
            text4 = font2.render(messages[i], True, RED)
            panel.blit(text4, [25, ypos])
            ypos = ypos + 18

        screen.blit(pygame.transform.rotate(panel, 90), (dimen1[1], 0))

    pygame.display.update()
    clock.tick(fps)

    if now.hour < WORK_HOUR[0] or now.hour > WORK_HOUR[1]:
        running = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

pygame.mouse.set_visible(True)
pygame.quit()
cap.release()
