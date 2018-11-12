from math import exp
import cv2.cv as cv
import numpy
from time import clock
from scipy.io import loadmat
import nao_2_0 as nao
#import csv
import sys
import os

## Find the *.mat file.
list_path = sys.path
for i in range (0,len(list_path)):
    if os.path.exists(list_path[i]+"/pythonNN2.mat"):
        break

mat = loadmat(list_path[i]+"/pythonNN2.mat")

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

def PitchYaw(image,intensity):
    global mat
    global image_list
    
    image_list= list()
    for k in range(0,20): # Convert image to list
        for l in range(0,45):
            image_list.append(image[l,k])
    image_list.append(intensity)

    image_mirrored = cv.CreateImage(cv.GetSize(image),image.depth,image.channels)

    cv.Copy(image,image_mirrored)

    cv.Flip(image_mirrored, flipMode = 1)
    cv.ShowImage("mirrored", image_mirrored)
    cv.ShowImage("small",image)
    key = cv.WaitKey(30)
    
    image_list_mirrored= list()
    for k in range(0,20): # Convert image to list
        for l in range(0,45):
            image_list_mirrored.append(image_mirrored[l,k])
    image_list_mirrored.append(intensity)

    outpitch = list()
    for i in range(1,17): # Calculate pitch for all available Networks
        if (i == 9) or (i == 10) or (i == 11) or (i == 12):
            if i > 9:
                name = str(i)
            else:
                name = '0'+str(i)
            start_time = clock()
            range_in = mat["pitch_0"+name+"_range_in"]
            w1 = mat["pitch_0"+name+"_w1"]
            w2 = mat["pitch_0"+name+"_w2"][0]
            rangenet = mat["pitch_0"+name+"_range"]
            b1 = mat["pitch_0"+name+"_b1"]
            b1 = remap(b1)
            b2 = mat["pitch_0"+name+"_b2"]
            b2 = remap(b2)

            image_ar = numpy.asarray(image_list)
            image_ar = mapminmax(image_ar*numpy.transpose((range_in[0][1]-range_in[0][0])/2))
            
            layer1out = tansig(numpy.dot(w1,image_ar)+b1)
            layer2out = numpy.dot(layer1out, w2)+b2;
            range_multi = (rangenet[0][1]-rangenet[0][0])/2;
            outpitch.append(range_multi*layer2out)

    outpitch_mirrored = list()
    for i in range(1,17): # Calculate pitch for all available Networks
        if (i == 9) or (i == 10) or (i == 11) or (i == 12):
            if i > 9:
                name = str(i)
            else:
                name = '0'+str(i)
            range_in = mat["pitch_0"+name+"_range_in"]    
            w1 = mat["pitch_0"+name+"_w1"]
            w2 = mat["pitch_0"+name+"_w2"][0]
            rangenet = mat["pitch_0"+name+"_range"]
            b1 = mat["pitch_0"+name+"_b1"]
            b1 = remap(b1)
            b2 = mat["pitch_0"+name+"_b2"]
            b2 = remap(b2)

            image_ar = numpy.asarray(image_list_mirrored)
            image_ar = mapminmax(image_ar*numpy.transpose((range_in[0][1]-range_in[0][0])/2))

            layer1out = tansig(numpy.dot(w1,image_ar)+b1)

            layer2out = numpy.dot(layer1out, w2)+b2;
            range_multi = (rangenet[0][1]-rangenet[0][0])/2;
            outpitch_mirrored.append(range_multi*layer2out)

    outyaw = list()
    for i in range(1,17):
        if (i != 5) or (i != 6) or (i != 7) or (i != 8):
            if i > 9:
                name = str(i)
            else:
                name = '0'+str(i)
            range_in = mat["yaw_0"+name+"_range_in"]
            w1 = mat["yaw_0"+name+"_w1"]
            w2 = mat["yaw_0"+name+"_w2"][0]
            rangenet = mat["yaw_0"+name+"_range"]
            b1 = mat["yaw_0"+name+"_b1"]
            b1 = remap(b1)
            b2 = mat["yaw_0"+name+"_b2"]
            b2 = remap(b2)
            
            image_ar = numpy.asarray(image_list)
            
            image_ar = mapminmax(image_ar*numpy.transpose((range_in[0][1]-range_in[0][0])/2))
            
            layer1out = tansig(numpy.dot(w1,image_ar)+b1)

            layer2out = numpy.dot(layer1out, w2)+b2;
            range_multi = (rangenet[0][1]-rangenet[0][0])/2;
            outyaw.append(range_multi*layer2out)

    outyaw_mirrored = list()
    for i in range(1,17):
        if (i != 5) or (i != 6) or (i != 7) or (i != 8):
            if i > 9:
                name = str(i)
            else:
                name = '0'+str(i)
                
            range_in = mat["yaw_0"+name+"_range_in"]
            w1 = mat["yaw_0"+name+"_w1"]
            w2 = mat["yaw_0"+name+"_w2"][0]
            rangenet = mat["yaw_0"+name+"_range"]
            b1 = mat["yaw_0"+name+"_b1"]
            b1 = remap(b1)
            b2 = mat["yaw_0"+name+"_b2"]
            b2 = remap(b2)
            
            image_ar = numpy.asarray(image_list_mirrored)
            image_ar = mapminmax(image_ar*numpy.transpose((range_in[0][1]-range_in[0][0])/2))

            layer1out = tansig(numpy.dot(w1,image_ar)+b1)

            layer2out = numpy.dot(layer1out, w2)+b2;
            range_multi = (rangenet[0][1]-rangenet[0][0])/2;
            outyaw_mirrored.append(range_multi*layer2out)

    outyaw = numpy.asarray(outyaw[0:len(outyaw)])
    outyaw_mirrored = numpy.asarray(outyaw_mirrored[0:len(outyaw)])
#    textfile.writerow([outyaw[0][0],outyaw[1][0],outyaw[2][0],outyaw[3][0],outyaw[4][0],outyaw[5][0],outyaw[6][0],outyaw[7][0],outyaw[8][0],outyaw[9][0],outyaw[10][0],outyaw[11][0],
#                       outyaw_mirrored[0][0],outyaw_mirrored[1][0],outyaw_mirrored[2][0],outyaw_mirrored[3][0],outyaw_mirrored[4][0],outyaw_mirrored[5][0],outyaw_mirrored[6][0],outyaw_mirrored[7][0],outyaw_mirrored[8][0],outyaw_mirrored[9][0],outyaw_mirrored[10][0],outyaw_mirrored[11][0]])
    outpitch = numpy.asarray(outpitch[0:len(outyaw)])
    outpitch_mirrored = numpy.asarray(outpitch_mirrored[0:len(outyaw)])

    return (numpy.mean(outpitch)+numpy.mean(outpitch_mirrored)/2,numpy.mean(outyaw)-numpy.mean(outyaw_mirrored)/2)

def HeadPose(image,region):
    
    global pitch_yaw
    global offset_pitch
    global offset_yaw
    global test
    save_time = clock()
    gray = cv.CreateImage(cv.GetSize(image),image.depth,1)

    if image.channels != 1:
        cv.CvtColor(image,gray,cv.CV_RGB2GRAY)
    else:
        cv.Copy(image,gray)    
    
    ## convert image to correct ratio
    ## the viola jones method returns a square region while we need a rectangular region
    region.x = region.x+(region.width*(1-0.6)/2)
    region.y = region.y + (region.height*(1-((90/40)*0.6))/2)
    region.height = region.height*(90/40)*0.6;
    region.width = region.width*0.6;

    ## filter image
    cv.SetImageROI(gray,(int(region.x),int(region.y),int(region.width),int(region.height)))
    intensity = float(cv.Avg(gray)[0])#calculate intensity here, otherwise info is lost
    small = cv.CreateImage((40,90),8,1)
    cv.Resize(gray,small)
    cv.Smooth(small,small,cv.CV_GAUSSIAN, 3)
    cv.EqualizeHist(small,small)
    lap_small_16 = cv.CreateImage(cv.GetSize(small),cv.IPL_DEPTH_16S,1)
    cv.Laplace(small,lap_small_16,3)
    laplace = cv.CreateImage(cv.GetSize(lap_small_16), 8,1) ## image number one
    cv.ConvertScaleAbs(lap_small_16,laplace)
    laplace_1 = cv.CreateImage((20,45),8,1)
    cv.Resize(laplace, laplace_1)
    del laplace
    laplace = laplace_1

    laplace3 = cv.CreateImage(cv.GetSize(laplace),laplace.depth,laplace.channels)
    cv.SetImageROI(laplace,(2,2,laplace.width-2,laplace.height-2))
    cv.Resize(laplace,laplace3) ## image number three slighty smaller image
    cv.ResetImageROI(laplace)

    ## Try tp determine the yaw and pitch using NN's    
    try:
        pitch_yaw1 = list(PitchYaw(laplace,intensity))
        pitch_yaw3 = list(PitchYaw(laplace3,intensity))

    except:
        print "PitchYaw function not working"
        
    pitch_yaw = list([(pitch_yaw1[0]+pitch_yaw3[0])/2,(pitch_yaw1[1]+pitch_yaw3[1])/2])  

    try:
        if draw:
            cv.ShowImage("test", image)
    except:
        pass
    k = cv.WaitKey(10)
    if k == 2490368:
        offset_pitch = offset_pitch+0.5
    elif k == 2621440:
        offset_pitch = offset_pitch-0.5
    elif k == 2555904:
        offset_yaw = offset_yaw+0.2
    elif k == 2424832:
        offset_yaw = offset_yaw-0.2

    return (-pitch_yaw[0]+offset_pitch, pitch_yaw[1]+offset_yaw)

if __name__ == "__main__":
    # This will start the main program. It uses the cam attached to the computer
    # to gather the images.
    capture = cv.CaptureFromCAM(0)
    storage = cv.CreateMemStorage()

    yaw_ar = list([0,0,0,0,0])
    pitch_ar = list([0,0,0,0,0])
    detected=False
#    textfile=csv.writer(open("test.csv","wb"))
    while True:
        image = cv.QueryFrame(capture)
        cv.ShowImage("image", image)
        key = cv.WaitKey(30)
        image, center, detected, region = nao.Detect(image, False)
        
        if detected:
            pitch, yaw = HeadPose(image, region)
            yaw_ar.append(-yaw)
            pitch_ar.append(pitch)
            yaw_ar.pop(0)
            pitch_ar.pop(0)
            print((yaw_ar[0]+yaw_ar[1]+yaw_ar[2]+yaw_ar[3]+yaw_ar[4])/5
                             ,(pitch_ar[0]+pitch_ar[1]+pitch_ar[2]+pitch_ar[3]+pitch_ar[4])/5)
        if key == 27:
            break

    cv.DestroyAllWindows()
    del capture
#    del test
#    del head
#    del textfile
    
