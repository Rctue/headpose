## This program needs input as two below mentioned HAAR cascades and pythonNN.mat to be present in list_path
## There is no other module that this program calls externally

from math import exp
import cv2.cv as cv
import cv2
import numpy
from time import clock
from scipy.io import loadmat
#import nao
import os
import sys
import glob
 
fast = 0
flip = 0
profile = 0


# cascade definition
cascade_folder = 'C:/development/opencv/sources/data/haarcascades/'
cascade_front = cv.Load(cascade_folder + "haarcascade_frontalface_alt2.xml")
cascade_profile = cv.Load(cascade_folder + "haarcascade_profileface.xml")

## Find the *.mat file.
list_path = sys.path
for i in range (0,len(list_path)):
    if os.path.exists(list_path[i]+"/pythonNN.mat"):
        break
 
mat = loadmat(list_path[i]+"/pythonNN.mat")
offset_pitch = 0
offset_yaw = 0
pitch_yaw = list([0,0])
 
outpitch = list()
outpitch_mirrored = list()
outyaw = list()
outyaw_mirrored = list()
 
class Region:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
 
def tansig(number):
    x = list()
    for i in range (0,len(number)):
        x.append((exp(number[i]) - exp(-number[i])) /(exp(number[i]) + exp(-number[i])))
    return numpy.asarray(x)
 
def mapminmax(array):
    difference = array.max() - array.min()
    array = array - (difference/2)
    array = array / (difference/2)
    return array
 
def remap(array):
    out = list()
    for i in range(0,len(array)):
        out.append(array[i][0])
    return numpy.asarray(out)
 
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
 
        image_ar = numpy.asarray(image_list)
        image_ar = mapminmax(image_ar)
         
        layer1out = tansig(numpy.dot(w1,image_ar)+b1)
 
        layer2out = numpy.dot(layer1out, w2)+b2;
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
 
        image_ar = numpy.asarray(image_list_mirrored)
        image_ar = mapminmax(image_ar)
 
        layer1out = tansig(numpy.dot(w1,image_ar)+b1)
 
        layer2out = numpy.dot(layer1out, w2)+b2;
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
         
         
        image_ar = numpy.asarray(image_list)
        image_ar = mapminmax(image_ar)
 
        layer1out = tansig(numpy.dot(w1,image_ar)+b1)
 
        layer2out = numpy.dot(layer1out, w2)+b2;
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
         
        image_ar = numpy.asarray(image_list_mirrored)
        image_ar = mapminmax(image_ar)
 
        layer1out = tansig(numpy.dot(w1,image_ar)+b1)
 
        layer2out = numpy.dot(layer1out, w2)+b2;
        range_multi = (rangenet[0][1]-rangenet[0][0])/2;

        outyaw_mirrored.append(range_multi*layer2out)


    # transfering all yaw & pitch values in respective arrays

    outyaw = numpy.asarray(outyaw[0:len(outyaw)])
    outyaw_mirrored = numpy.asarray(outyaw_mirrored[0:len(outyaw_mirrored)])
    outpitch = numpy.asarray(outpitch[0:len(outpitch)])
    outpitch_mirrored = numpy.asarray(outpitch_mirrored[0:len(outpitch_mirrored)])

    # median is considered to remove influence caused by outliers with mean
    return (numpy.mean(outpitch)+numpy.mean(outpitch_mirrored))/2,(numpy.median(outyaw)-numpy.median(outyaw_mirrored))/2


######################## Detect function detects faces using HAAR cascades ########################

def Detect(frame, draw = True):
    global face1_x
    global face1_y
    global face1_width
    global face1_center
    global old_face1_x
    global old_face1_y
    global old_face1_width
    global fast
    global windowsz
    global cascade_front
    global flip
    global profile
    
    roiscale = 2
    windowscale = 10
    face1_center = (0,0)
    flip=0  # if flipped 
    profile=0   # if detected by frontal (profile = 0) or profile (profile = 1) cascade
    

    frame_gray = cv.CreateImage(cv.GetSize(frame),frame.depth,1)
    frame_gray_flipped = cv.CreateImage(cv.GetSize(frame),frame.depth,1)

    windowsz = 30

    if frame.channels != 1:
        cv.CvtColor(frame,frame_gray,cv.CV_RGB2GRAY)    # RGB 3 channel to 1 channel gray image conversion
    else:
        cv.Copy(frame,frame_gray)  

    cv.EqualizeHist(frame_gray,frame_gray)      # histogram equalization before cascading

    # HaarDetectObjects() uses fixed properties for all cascades 
    # scaleFactor = 1.3
    # min neighbours = 4
    # flags = CV_HAAR_SCALE_IMAGE
    # (min,max) possible object size = (30,30)

    # frontal HAAR cascade on normal gray image
    faces = cv.HaarDetectObjects(frame_gray, cascade_front, storage, 1.3, 4, 0|cv.CV_HAAR_SCALE_IMAGE, (windowsz,windowsz))
    if faces:
        #print "front"
        flip=0
        profile=0
    else:
        cv.Flip(frame_gray, frame_gray_flipped, 1)
        cv.EqualizeHist(frame_gray_flipped,frame_gray_flipped)

    # frontal HAAR cascade on flipped gray image
    # 'faces' contains co-ordinates of the bounding box (region) on the mirrored image, not the actual one
    faces = cv.HaarDetectObjects(frame_gray_flipped, cascade_front, storage, 1.3, 4, 0|cv.CV_HAAR_SCALE_IMAGE, (windowsz,windowsz))
    if faces:
        flip = 1
        #print "flipped front"
        profile=0   
    
    else:
        # profile HAAR cascade on normal gray image
        faces = cv.HaarDetectObjects(frame_gray, cascade_profile,storage, 1.3, 4, 0|cv.CV_HAAR_SCALE_IMAGE, (windowsz,windowsz))
        if faces:
            #print "profile"
            flip=0
            profile=1
        else:
            # profile HAAR cascade on flipped gray image
            faces = cv.HaarDetectObjects(frame_gray_flipped, cascade_profile, storage, 1.3, 4, 0|cv.CV_HAAR_SCALE_IMAGE, (windowsz,windowsz))
            if faces:           
                flip=1  
                #print "flipped profile"
                profile=1
            else:
                flip=0
    
    
    cv.ResetImageROI(frame)  
 
    try:
    
        face1_x = faces[0][0][0] 
        face1_y = faces[0][0][1]

        face1_width = faces[0][0][2]
        face1_height = faces[0][0][3]
        face1_center = (face1_x + (face1_width/2),face1_y + (face1_height/2))

        region = Region()
        region.x = face1_x
        region.y = face1_y
        region.width = face1_width
        region.height = face1_height


        if draw == True:
            if flip == 0:
                cv.Rectangle(frame, (face1_x, face1_y),
                            (face1_x+face1_width,face1_y+face1_height),
                            cv.RGB(255,255,255))
                cv.Circle(frame, face1_center, 2, cv.RGB(255, 0, 0))
            else:
                #calculate the position of actual face from mirrored co-ordinates
                cv.Rectangle(frame, (frame.width-face1_x-face1_width, face1_y), 
                                    (frame.width-face1_x, face1_y+face1_height),
                            cv.RGB(255,255,255))
                face1_center = (frame.width-face1_x-face1_width + (face1_width/2),face1_y + (face1_height/2))
                cv.Circle(frame, face1_center, 2, cv.RGB(255, 0, 0))

    except:
        region = Region()

    if faces:
        facedetected = True
    else:
        facedetected = False

    face_loc = list(face1_center)
    convrad = 0.55/(frame.width/2)
    face_loc[0] = (face_loc[0] - (frame.width/2))*convrad
    face_loc[1] = (face_loc[1] - (frame.height/2))*convrad
    
    return frame, face_loc, flip, profile, facedetected, region 


######################## HeadPose calculation function to call PitchYaw function ########################

def HeadPose(image, isFlipped, region):
     
    global pitch_yaw
    global offset_pitch
    global offset_yaw
    
    gray = cv.CreateImage(cv.GetSize(image),image.depth,1)
 
    if image.channels != 1:
        cv.CvtColor(image,gray,cv.CV_RGB2GRAY)
    else:
        cv.Copy(image,gray)    
    
    if isFlipped == 1:      # flipping operation needed if an image is detected after flipping before passing to NN
        cv.Flip(gray, gray, 1)
    
    ## convert image to correct ratio (40:90)
    ## the viola jones method returns a square region while we need a rectangular region
    region.x = region.x+(region.width*(1-0.6)/2)
    region.y = region.y + (region.height*(1-((90/40)*0.6))/2)
    region.height = region.height*(90/40)*0.6;
    region.width = region.width*0.6;
    
    ## filter image (laplace)
    cv.SetImageROI(gray,(int(region.x),int(region.y),int(region.width),int(region.height)))
    small = cv.CreateImage((40,90),8,1)
    cv.Resize(gray,small)
    cv.Smooth(small,small,cv.CV_GAUSSIAN, 3)
    cv.EqualizeHist(small,small)
   
    lap_small_16 = cv.CreateImage(cv.GetSize(small),cv.IPL_DEPTH_16S,1)
    cv.Laplace(small,lap_small_16,3)
    laplace = cv.CreateImage(cv.GetSize(lap_small_16), 8,1) ## image number one
    cv.ConvertScaleAbs(lap_small_16,laplace)

    laplace3 = cv.CreateImage(cv.GetSize(laplace),laplace.depth,laplace.channels)
    cv.SetImageROI(laplace,(2,2,laplace.width,laplace.height))
    cv.Resize(laplace,laplace3) ## image number three slighty smaller image

    cv.ResetImageROI(laplace)
 
    ## Try to determine the yaw and pitch using NN's    
    try:
        pitch_yaw=list(PitchYaw(laplace))       # filtered and scaled image passes to NN

    except:
        print "PitchYaw function not working"

    return (pitch_yaw[0], pitch_yaw[1])         # returned pitch and yaw values from NN


######################## main function ########################

if __name__ == "__main__":
    
##    for filename in glob.glob('/home/suvadeep/Desktop/headpose/faces_point_updated/*.jpg'):
##    
##    #filename = '/home/suvadeep/Desktop/headpose/faces_point_updated/personne10242+0-60.jpg'   # individual image check
##
##        pitch_yaw_actual = filename[65:-4]  # this substring extracts pitch & yaw from file name and depends on the file location
##        fileName = filename[60:71]
##
##        image = cv.LoadImage(filename)
    capture = cv.CaptureFromCAM(0)
    storage = cv.CreateMemStorage()
    detected=False
    while True:
        image = cv.QueryFrame(capture)

        image, center, isFlipped, isProfile, detected, region = Detect(image, False)
        cv.ShowImage("image", image)

        if detected:
            pitch, yaw = HeadPose(image, isFlipped, region)     # detected region and flip condition passed to HeadPose()
    
            if isFlipped == 0:
                yaw=-yaw                        # real head orientation in computer display is opposite.
#                    print ("%.3f\t %.3f\t\t %s\t %d\t %d\t %s" % (float(pitch), float(yaw), pitch_yaw_actual, isProfile, isFlipped, fileName))
            else:
#                print ("%.3f\t %.3f\t\t %s\t %d\t %d\t %s" % (float(pitch), float(yaw), pitch_yaw_actual, isProfile, isFlipped, fileName))
                pass
            print ("%.3f\t %.3f\t\t %d\t %d" % (float(pitch), float(yaw), isProfile, isFlipped))

    
        key=cv.WaitKey(10)
        if key == 27:
            break       
     
    cv.DestroyAllWindows()
