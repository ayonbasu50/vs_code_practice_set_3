# import os
# print(os.path.abspath("."))

# pylint:disable=no-member

import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('opencv-course-master\Section3 - Faces\haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
features = np.load('opencv-course-master\\Section3 - Faces\\features.npy', allow_pickle=True)
labels = np.load('opencv-course-master\\Section3 - Faces\\labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('opencv-course-master\\Section3 - Faces\\face_trained1.yml')

img = cv.imread('opencv-course-master\\Resources\\Faces\\val\\mindy_kaling\\3.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 6)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
