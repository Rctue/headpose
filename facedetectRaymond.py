# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2

cv2_window_name = "Image window 1"
cascade_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cascade_front = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
cascade_profile = cv2.CascadeClassifier('haarcascade_profileface.xml')

class Region:
    def __init__(self,x=0,y=0,width=0,height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def Detect(frame, draw_rectangles = True):
    global cascade_default
    
    #capture frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale to improve detection speed and accuracy
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray) # equalize contrast histogram

    #Run classifier on frame
    faces = cascade_default.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    face_list=[]
    if len(faces) >= 1: #Use simple check if one face is detected, or multiple (measurement error unless multiple persons on image)
        for (x, y, w, h) in faces:
            #faceslice = gray[y:y + h, x:x + w] #slice face from image
            if draw_rectangles:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) #draw rectangle around face in "frame", (coordinates), (size), (RGB color), thickness 2
            face_list.append(Region(x,y,w,h))
        detected = True
        #cv2.imshow("detect", faceslice) #display sliced face
    else:
        detected = False
        #print("no/multiple faces detected, passing over frame")

    return detected, face_list, frame
    
def facerecog(show_image=False):
    video_capture = cv2.VideoCapture(0)

    done=False
    while not done:
        ret, frame = video_capture.read() #Grab frame from webcam. Ret is 'true' if the frame was successfully grabbed.
        detected, face_list, frame = Detect(frame) #Grab frame from webcam. Ret is 'true' if the frame was successfully grabbed.
        if show_image:
            cv2.imshow(cv2_window_name, frame) #Display frame
            if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition to work correctly, here it is bound to key 'q'
                cv2.destroyWindow(cv2_window_name)
                done=True
        else:
            done=True
    if detected:
        return face_list[0], detected
    else:
        return None, detected

if __name__=="__main__":
    region, detected = facerecog(True)
    if detected:
        print region.x, region.y, region.width, region.height
    else:
        print "no face detected"
