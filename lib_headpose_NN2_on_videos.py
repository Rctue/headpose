from math import exp
import cv2
import numpy
from time import clock
from scipy.io import loadmat
import facedetectRaymond
import csv
import sys
import os

import shutil
from imutils.video import FPS
import time

def stop(fileName, csvData, csvNoFace, fpsArray, elapsed, fps, startTime):
    # pathVideo = "D:\Onderzoek\\videosWithHeadpose\\"
    # shutil.copy('videoHeadpose.avi', pathVideo)
    # os.rename(pathVideo+ "videoHeadpose.avi", pathVideo + fileName + ".avi")

    #text_file.close()
    pathHeadpose = "D:\Onderzoek\\baselineAlgoritme\\\headposeData"

    timestamp = timer(time.time() - startTime)
    fpsArray.append([fileName, "{:.2f}".format(elapsed), timestamp,"{:.2f}".format(fps)])

    with open('data.csv', 'wb') as csvFile:
        writer = csv.writer(csvFile, delimiter=";")
        writer.writerows(csvData)
        csvFile.close()
    shutil.copy('data.csv', pathHeadpose)
    # for i in range(60):
    #     if Path(path + "\\data" + str(i+1) + ".csv").is_file() == False:
    #         fileName = "data" + str(i+1)
    #         break
    os.rename(pathHeadpose+"\\data.csv",pathHeadpose + "\\headpose-" + fileName + ".csv")

    pathNoFace = "D:\Onderzoek\\baselineAlgoritme\\noFaceData"
    with open('noFaceData.csv', 'wb') as csvFile:
        writer = csv.writer(csvFile, delimiter=";")
        writer.writerows(csvNoFace)
        csvFile.close()

    shutil.copy('noFaceData.csv', pathNoFace)
    os.rename(pathNoFace+"\\noFaceData.csv",pathNoFace + "\\noFace-" + fileName + ".csv")

def msConverter(miliSeconds):
    seconds = (miliSeconds/1000)%60
    minutes = (miliSeconds/(1000*60))%60
    frameTimestamp = "%d:%f" % (minutes, seconds)
    return frameTimestamp

def timer(sec):
  mins = sec // 60
  sec = sec % 60
  mins = mins % 60
  time = ("{0}:{1}".format(int(mins),sec))
  return time

## Find the *.mat file.
list_path = sys.path
for i in range (0,len(list_path)):
    if os.path.exists(list_path[i]+"/pythonNN2.mat"):
        break

mat = loadmat(list_path[i]+"/pythonNN2.mat")

draw = True #W: Ik heb dit zelf toegevoegd, omdat de hele code anders niet werkte

class Region:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

def CreateImage(size, channels=1, bits = numpy.uint8):
    if channels > 1:
        image = numpy.zeros((size[0], size[1], channels), bits)
    else:
        image = numpy.zeros(size, bits)
    return image

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
    
    image_list= []
    for k in range(0,20): # Convert image to list
        for l in range(0,45):
            image_list.append(image[l,k])
    image_list.append(intensity)

    image_mirrored = cv2.flip(image, 1)
    #cv2.imshow("mirrored", image_mirrored)
    #cv2.imshow("small",image)
    #W: deze twee imshows laten geen beelden zien

    #key = cv2.waitKey(0)
    
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

    #gray = CreateImage(image.shape, 1)

    if image.ndim >2:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    ## convert image to correct ratio
    ## the viola jones method returns a square region while we need a rectangular region
    region.x = region.x+(region.width*(1-0.6)/2)
    region.y = region.y + (region.height*(1-((90/40)*0.6))/2)
    region.height = region.height*(90/40)*0.6;
    region.width = region.width*0.6;

    ## filter image
    #cv2.selectROI(gray, (int(region.x), int(region.y), int(region.width), int(region.height)))
    intensity = float(numpy.average(gray))#calculate intensity here, otherwise info is lost
    #small = CreateImage((40,90),1)
    small = gray[int(region.y):int(region.y + region.height),int(region.x):int(region.x + region.width)]
    small = cv2.GaussianBlur(small, (3, 3), cv2.BORDER_DEFAULT)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    small = clahe.apply(small)
    #W: Tot hier gaat alles nog goed, en zien all beelden er goed uit
    cv2.imshow("gray", small)
    
    
    lap_small_16 = cv2.Laplacian(small,cv2.CV_16S, ksize=3)
    #W: In de bovenste twee regels gaat iets fout waardoor er geen beeld meer is
    #W: dit is volgens mij de reden dat de hele code niet meer werkt

    laplace = cv2.convertScaleAbs(lap_small_16)
    laplace = cv2.resize(laplace, (20, 45))
    laplace3 = laplace[2:laplace.shape[0]-2, 2:laplace.shape[1]-2]
    #cv2.SetImageROI(laplace,(2,2,laplace.width-2,laplace.height-2))

    laplace3 = cv2.resize(laplace3, (20, 45)) ## image number three slighty smaller image
    #cv2.ResetImageROI(laplace)
    #ROI is lastig aan te passen en om te schrijven

    #print laplace.shape
    #cv2.imshow("laplace", laplace)
   

    ## Try tp determine the yaw and pitch using NN's
    #try:
    pitch_yaw1 = PitchYaw(laplace,intensity)
    pitch_yaw3 = PitchYaw(laplace3,intensity)
    #W: om de code te testen moet de try gecomment zijn, anders gaat hij altijd op de except function

    #except:
     #   print("PitchYaw function not working")
        
    pitch_yaw = [(pitch_yaw1[0]+pitch_yaw3[0])/2,(pitch_yaw1[1]+pitch_yaw3[1])/2]  

##    try:
##        if draw:
##            #W: Ik heb geen idee waar normaal gesproken draw vandaan komt
##            #W: Het stond/staat verder nergens in de originele code
##            #W: Dus dit zou altijd niet moeten runnen en naar except gaan
##            cv2.imshow("draw", image)
##            #W: nogsteeds geen beeld
##    except:
##        pass
    k = cv2.waitKey(10)
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
    fpsArray = [["video", "elapsedTime", "time", "fps"]]

    videos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20,
              21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

    for video in videos:
        startTime = time.time()
        csvData = [['yaw', 'pitch', 'timestamp']]
        csvNoFace = [["time", "random"]]

        offset_pitch = 0
        offset_yaw = 0
        pitch_yaw = list([0,0])

        outpitch = list()
        outpitch_mirrored = list()
        outyaw = list()
        outyaw_mirrored = list()

        fileName = "video"+str(video)
        VIDEO_SRC = "D:\\Onderzoek\\\edittedVideos\\"+fileName+".avi"
        capture = cv2.VideoCapture(VIDEO_SRC)
        fps = FPS().start()
        yaw_ar = [0, 0, 0, 0, 0]
        pitch_ar = [0, 0, 0, 0, 0]

        while True:
            detected = False

            ret, frame = capture.read()
            msFrame = msConverter(capture.get(cv2.CAP_PROP_POS_MSEC))
            fps.update()

            if not ret:
                fps.stop()
                stop(fileName, csvData, csvNoFace, fpsArray, fps.elapsed(), fps.fps(), startTime)
                break

            #print("test2")
            detected, face_list, image = facedetectRaymond.Detect(frame, False)

            if detected:
                #print("test3")
                pitch, yaw = HeadPose(frame, face_list[0])
                yaw_ar.append(-yaw)
                pitch_ar.append(pitch)
                yaw_ar.pop(0)
                pitch_ar.pop(0)
                yaw = (yaw_ar[0]+yaw_ar[1]+yaw_ar[2]+yaw_ar[3]+yaw_ar[4])/5
                pitch = (pitch_ar[0]+pitch_ar[1]+pitch_ar[2]+pitch_ar[3]+pitch_ar[4])/5
                print((yaw_ar[0]+yaw_ar[1]+yaw_ar[2]+yaw_ar[3]+yaw_ar[4])/5
                                 ,(pitch_ar[0]+pitch_ar[1]+pitch_ar[2]+pitch_ar[3]+pitch_ar[4])/5)
                currentData = [yaw, pitch, msFrame]
                csvData.append(currentData)
                #W: Het is verbazend dat hier iets geprint wordt, alleen het is altijd hetzelfde
                #W: Dit is logisch want er is geen beeld om iets mee te scannen
            else:
                frame = cv2.flip(frame,1)
                detected, face_list, image = facedetectRaymond.Detect(frame, False)
                if detected:
                    # print("test3")
                    pitch, yaw = HeadPose(frame, face_list[0])
                    yaw_ar.append(yaw)
                    pitch_ar.append(pitch)
                    yaw_ar.pop(0)
                    pitch_ar.pop(0)
                    print((yaw_ar[0] + yaw_ar[1] + yaw_ar[2] + yaw_ar[3] + yaw_ar[4]) / 5
                          , (pitch_ar[0] + pitch_ar[1] + pitch_ar[2] + pitch_ar[3] + pitch_ar[4]) / 5)
                    yaw = (yaw_ar[0] + yaw_ar[1] + yaw_ar[2] + yaw_ar[3] + yaw_ar[4]) / 5
                    pitch = (pitch_ar[0] + pitch_ar[1] + pitch_ar[2] + pitch_ar[3] + pitch_ar[4]) / 5
                    currentData = [yaw, pitch, msFrame]
                    csvData.append(currentData)

                else:
                    csvNoFace.append([str(msFrame), 1])

            #cv2.imshow("image", frame)
            key = cv2.waitKey(10)
            if key & 0xFF == 27:
                fps.stop()
                stop(fileName, csvData, csvNoFace, fpsArray, fps.elapsed(), fps.fps(), startTime)
                break
        cv2.destroyAllWindows()
        del capture

    pathFps = "D:\Onderzoek\\baselineAlgoritme\FPSdata"
    with open('fpsData.csv', 'wb') as csvFile:
        writer = csv.writer(csvFile, delimiter=";")
        writer.writerows(fpsArray)
        csvFile.close()

    shutil.copy('fpsData.csv', pathFps)
    os.rename(pathFps + "\\fpsData.csv", pathFps + "\\dataFps.csv")

#    del test
#    del head
#    del textfile
    
