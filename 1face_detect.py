#pylint:disable=no-member

# create haar_face.xml from opencv/data/haarcascades/ haarcascade_frontalface_default.xml


import cv2 as cv

img = cv.imread('opencv-course-master\Resources\Photos\group 1.jpg')
cv.imshow('Group of 5 people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

haar_cascade = cv.CascadeClassifier('opencv-course-master\Section3 - Faces\haar_face.xml')

# change the minNeighbors=3 depending in person in image

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)



cv.waitKey(0)