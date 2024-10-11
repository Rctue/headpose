# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

class Region:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

def facerecog():
    video_capture = cv2.VideoCapture(0)
    facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    region = Region()
    if True:
        ret, frame = video_capture.read() #Grab frame from webcam. Ret is 'true' if the frame was successfully grabbed.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale to improve detection speed and accuracy
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        #Run classifier on frame
        face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in face: #Draw rectangle around detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) #draw it on "frame", (coordinates), (size), (RGB color), thickness 2
        if len(face) == 1: #Use simple check if one face is detected, or multiple (measurement error unless multiple persons on image)
            for (x, y, w, h) in face:
                faceslice = gray[y:y + h, x:x + w] #slice face from image
                region.x = x
                region.y = y
                region.width = w
                region.height = h
            detected = True
            cv2.imshow("detect", faceslice) #display sliced face
        else:
            detected = False
            print("no/multiple faces detected, passing over frame")
        cv2.imshow("webcam", frame) #Display frame
        #if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition to work correctly, here it is bound to key 'q'
        #    break
    return region, detected

region, detected = facerecog()
print(detected)
print(region)