## This program needs input as two below mentioned HAAR cascades and pythonNN.mat to be present in list_path
## There is no other module that this program calls externally

#import cv2.cv as cv
import cv2
import numpy as np
from scipy.io import loadmat
import facedetect

#from math import exp
#from time import clock
#import nao
import os
import sys
 
#globals
fast = 0

def load_NN(file_name='pythonNN.mat'):
    ## Find the *.mat file.
    for p in sys.path:
        if os.path.exists(p + "/" + file_name):
            return loadmat(p + "/" + file_name)
    return None

mat = load_NN()
offset_pitch = 0
offset_yaw = 0
pitch_yaw = [0,0]
 
outpitch = list()
outpitch_mirrored = list()
outyaw = list()
outyaw_mirrored = list()
 
class Region:
    def __init__(self, x=0,y=0,w=0,h=0):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
 
def tansig(number):
    x = list()
    for i in range (0,len(number)):
        x.append((np.exp(number[i]) - np.exp(-number[i])) /(np.exp(number[i]) + np.exp(-number[i])))
    return np.asarray(x)
 
def mapminmax(array):
    difference = array.max() - array.min()
    array = array - (difference/2)
    array = array / (difference/2)
    return array
 
def remap(array):
    out = list()
    for i in range(0,len(array)):
        out.append(array[i][0])
    return np.asarray(out)
 
def CreateImage(size, channels, bits = np.uint8):
    if channels > 1:
        image = np.zeros((size[0], size[1], channels), bits)
    else:
        image = np.zeros(size, bits)
    return image

######################## PitchYaw function ########################

def PitchYaw(image):
    global mat
    global image_list
    global outpitch
    global outpitch_mirrored
    global outyaw
    global outyaw_mirrored
 
    image_list= list()
    for i in range(0,90): # Convert image to list
        for j in range(0,40):
            image_list.append(image[i,j])
 
    image_list_mirrored= list()
    for i in range(0,90): # Convert image to list
        for j in range(39,-1,-1):
            image_list_mirrored.append(image[i,j])
 
    ## calculation of pitch and yaw using NN

    outpitch = list()
    for i in range(1,11): # Calculate pitch for all available Networks
        if i > 9:
            name = str(i)
        else:
            name = '0'+str(i)

        w1 = mat["pitch_0"+name+"_w1"]
        w2 = mat["pitch_0"+name+"_w2"][0]
        rangenet = mat["pitch_0"+name+"_range"]
        b1 = mat["pitch_0"+name+"_b1"]
        b1 = remap(b1)
        b2 = mat["pitch_0"+name+"_b2"]
        b2 = remap(b2)
 
        image_ar = np.asarray(image_list)
        image_ar = mapminmax(image_ar)
         
        layer1out = tansig(np.dot(w1,image_ar)+b1)
 
        layer2out = np.dot(layer1out, w2)+b2;
        range_multi = (rangenet[0][1]-rangenet[0][0])/2;

        outpitch.append(range_multi*layer2out)

 
    outpitch_mirrored = list()
    for i in range(1,11): # Calculate pitch for all available Networks
        if i > 9:
            name = str(i)
        else:
            name = '0'+str(i)
             
        w1 = mat["pitch_0"+name+"_w1"]
        w2 = mat["pitch_0"+name+"_w2"][0]
        rangenet = mat["pitch_0"+name+"_range"]

        b1 = mat["pitch_0"+name+"_b1"]
        b1 = remap(b1)
        b2 = mat["pitch_0"+name+"_b2"]
        b2 = remap(b2)
 
        image_ar = np.asarray(image_list_mirrored)
        image_ar = mapminmax(image_ar)
 
        layer1out = tansig(np.dot(w1,image_ar)+b1)
 
        layer2out = np.dot(layer1out, w2)+b2;
        range_multi = (rangenet[0][1]-rangenet[0][0])/2;

        outpitch_mirrored.append(range_multi*layer2out)

 
    outyaw = list()
    for i in range(1,11): # Calculate yaw for all available Networks
        if i > 9:
            name = str(i)
        else:
            name = '0'+str(i)
 
        w1 = mat["yaw_0"+name+"_w1"]
        w2 = mat["yaw_0"+name+"_w2"][0]
        rangenet = mat["yaw_0"+name+"_range"]
        b1 = mat["yaw_0"+name+"_b1"]
        b1 = remap(b1)
        b2 = mat["yaw_0"+name+"_b2"]
        b2 = remap(b2)
         
         
        image_ar = np.asarray(image_list)
        image_ar = mapminmax(image_ar)
 
        layer1out = tansig(np.dot(w1,image_ar)+b1)
 
        layer2out = np.dot(layer1out, w2)+b2;
        range_multi = (rangenet[0][1]-rangenet[0][0])/2;

        outyaw.append(range_multi*layer2out)
 
 
    outyaw_mirrored = list()
    for i in range(1,11): # Calculate yaw for all available Networks
        if i > 9:
            name = str(i)
        else:
            name = '0'+str(i)
             
        w1 = mat["yaw_0"+name+"_w1"]
        w2 = mat["yaw_0"+name+"_w2"][0]
        rangenet = mat["yaw_0"+name+"_range"]
        b1 = mat["yaw_0"+name+"_b1"]
        b1 = remap(b1)
        b2 = mat["yaw_0"+name+"_b2"]
        b2 = remap(b2)
         
        image_ar = np.asarray(image_list_mirrored)
        image_ar = mapminmax(image_ar)
 
        layer1out = tansig(np.dot(w1,image_ar)+b1)
 
        layer2out = np.dot(layer1out, w2)+b2;
        range_multi = (rangenet[0][1]-rangenet[0][0])/2;

        outyaw_mirrored.append(range_multi*layer2out)


    # transfering all yaw & pitch values in respective arrays

    outyaw = np.asarray(outyaw[0:len(outyaw)])
    outyaw_mirrored = np.asarray(outyaw_mirrored[0:len(outyaw_mirrored)])
    outpitch = np.asarray(outpitch[0:len(outpitch)])
    outpitch_mirrored = np.asarray(outpitch_mirrored[0:len(outpitch_mirrored)])

    # median is considered to remove influence caused by outliers with mean
    return (np.mean(outpitch)+np.mean(outpitch_mirrored))/2,(np.median(outyaw)-np.median(outyaw_mirrored))/2


######################## Detect function detects faces using HAAR cascades ########################

def Detect(frame, draw = True):
    
    #frame_gray_flipped = CreateImage(frame.shape,1)

    if frame.ndim > 2:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray=frame.copy()  

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(frame_gray) # equalize contrast histogram
    #equalized =cv2.equalizeHist(frame_gray)      # histogram equalization before cascading
    eq_flipped = cv2.flip(equalized, 1)

    # frontal HAAR cascade on normal gray image
    face_detected = False
    flip = None
    profile = None
    faces = facedetect.cascade_profile.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces)>=1:
        face_detected = True
        flip = False
        profile = False
    else:
        faces = facedetect.cascade_front.detectMultiScale(eq_flipped, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces)>=1:
            face_detected = True
            flip = True
            profile = False
        else:
            faces = facedetect.cascade_profile.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces)>=1:
                face_detected = True
                flip = False
                profile = True
            else:
                faces = facedetect.cascade_profile.detectMultiScale(eq_flipped, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces)>=1:
                    face_detected = True
                    flip = True
                    profile = True
    # faces can still be empty

    if face_detected:
        for i in range(len(faces)):
            (x, y, w, h) = faces[i] 
            #face1_center = (face1_x + (face1_width/2),face1_y + (face1_height/2))
            if flip == True:
                x=frame_gray.shape[1]-x-w
                faces[i][3]=x
                cv2.putText(frame,'Flipped',(x,y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,25), 1, cv2.LINE_AA) #print "flipped detection"
            if profile == True:
                cv2.putText(frame,'Profile',(x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,25), 1, cv2.LINE_AA) #print "flipped detection"
                    
            if draw == True:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2) #image, start_point, end_point, BGR and thickness
                #cv2.circle(frame, (x+w/2,y+h/2), 2, (255, 0, 0))
                #print x,y,w,h

    return face_detected, faces, frame_gray

def compute_face_centers(face_list, im_shape):
    convrad = 0.55/(frame.shape[0]/2)  # radians/pixel, 0.55: half the image width in radians (31.5 degrees)
    face_loc_list=[]
    for (x, y, w, h) in face_list: 
        face_loc = [float(x + w/2)*convrad, float(y + h/2)*convrad]
        face_loc_list.append(face_loc)
    
    return face_loc_list
######################## HeadPose calculation function to call PitchYaw function ########################

def HeadPose(image, (x,y,w,h)):
     
    global pitch_yaw
    global offset_pitch
    global offset_yaw
    
    if image.ndim == 3:
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()    
        
    ## convert image to correct ratio (40:90)
    ## the viola jones method returns a square region while we need a rectangular region
    rw = int(w*0.6)
    rh = int(9.0/4.0*w*0.6) # note that w=h
    rx = int(x+(1.0 - 0.6)*w/2.0)
    ry = int(y+(1.0 - 0.6*9.0/4.0)*w/2.0)

    ## filter image (laplace)
    gray_small = gray[ry:ry+rh, rx:rx+rw]    
    
    small = cv2.resize(gray_small,(40,90))
    small = cv2.GaussianBlur(small, (3, 3), cv2.BORDER_DEFAULT)
    small = cv2.equalizeHist(small)
    
    laplace = cv2.Laplacian(small, cv2.CV_16S, ksize=3)
    laplace = cv2.convertScaleAbs(laplace)
    #laplace = laplace[2:laplace.shape[1]-2, 2:laplace.shape[0]-2] # crop edges
    cv2.imshow("detect image", laplace)
 #   cv2.waitKey(0)
    
    
    ## Try to determine the yaw and pitch using NN's    
 #   try:
    pitch_yaw=PitchYaw(laplace)       # filtered and scaled image passes to NN

    #except:
    #    print("PitchYaw function not working")

    return (pitch_yaw[0], pitch_yaw[1])         # returned pitch and yaw values from NN


######################## main function ########################

if __name__ == "__main__":
    device=0 # default video device
    capture = cv2.VideoCapture(device)  
    
    
    done=False
    detected=False
    while not done:
        ret, frame = capture.read()
        
        if ret:
            detected, face_list, image = Detect(frame, True)
            face_loc_list = compute_face_centers(face_list, image.shape)
            
            if detected:
                #for face in face_list:
                #    pitch, yaw = HeadPose(image, face)     # detected face and flip condition passed to HeadPose()
                pitch, yaw = HeadPose(image, face_list[0])
                print ("pitch %.3f\tyaw %.3f" % (float(pitch), float(yaw)))
    
        
            cv2.imshow("detected image", frame)
            key=cv2.waitKey(10)
            if key & 0xFF == 27:
                done=True
                break
     
    cv2.destroyAllWindows()
