## Nao functions version 1.30
## change log:
## 1.02: Added class "Region"
## 1.02: Resolution stuff.
## 1.03: Detect() now returns an object of the class "Region()"
## 1.04: Added Aldebarans face detection NaoFaceLocation().
## 1.05: Added the gesture() function and EyeLED() function.
## 1.06: Now able to look for the correct haarcascade file within the pythonpath
## 1.07: Changed Track() function to better support different frame rates
## 1.08: Added ALTrack function
## 1.09: Added second gesture in Gesture()
## 1.10: Added InitPose
## 1.11: Added Move
## 1.12: Added Crouch
## 1.13: Removed Gesture(), instead use the gesture lib. Changed comments for Move()
## 1.14: Added Play() function for playing sound files
## 1.15: Added Record() function
## 1.16: Added WalkTo function
## 1.17: Added PlaySine function
## 1.18: Added function FindFace()
## 1.19: Added RunMovement() (19-09-2011 - Turin)
## 1.20: Added Stiffen() for stiffening the joints
## 1.21: Added RunLed() for running led scripts
## 1.22: GetAvailableLEDPatterns() and GetAvailableGestures() added.
## 1.23: speechProxy added
## 1.24: File existence check added in RunMovement, RunLed, RunSpeech
## 1.25: Fixed remove = remove.reverse() returning None error
## 1.26: Added InitSpeech() and DetectSpeech()
## 1.27: GetAvailableDialogs() added.
## 1.28: Added LoadDialog()
## 1.29: Changed searchpaths of RunLED, RunMovement and RunSpeech to include /led, /gestures and /tts subfolders, respectively.
## 1.30: Added possibility of sending port number to InitProxy
## 1.31: Added better error handling in several functions and made posting of text optional.
## 1.32: RunLED changed to read files with ; as delimiter and to deal with multi-line led-files
## 1.33: LoadDialog() reads files with ; as delimiter
## 1.34: Added functions MoveHead() to move nao's head and GetYaw() to request the yaw of nao's head 
## 1.35: Added functions SetTTSVolume() and GetTTSVolume() for checking and controlling the volume of the Text to Speech
## 1.36: Added functions SetMusicVolume() and GetMusicVolume() for checking and controlling the volume of the Music

### nao_nocv.py
## 1.0 removed dependencies on OpenCV and Image libraries. InitVideo and GetImage modified, but broken.

#import cv
from time import time
from time import sleep
#import Image
import random
import math
import sys
import os
import csv
import naoqi
from collections import deque

gftt_list = list()
fast = 0
time_q = deque([1,1,1,1,1,1,1,1,1,1])
old_time = time()
time_old_track = time()
#font = cv.InitFont(cv.CV_FONT_HERSHEY_TRIPLEX, 0.5, 0.5, 0.0, 1)

## Find the *.xml file for face detection.
list_path = sys.path
for i in range (0,len(list_path)):
    if os.path.exists(list_path[i]+"/haarcascade_frontalface_alt2.xml"):
        break
   
#cascade_front = cv.Load(list_path[i]+"/haarcascade_frontalface_alt2.xml")

interpol_time=0.3
start_mov_t = time()
weights = list()
existence = list()
id_pose = None
alface_subscribed = False
xtargetold = 0
ytargetold = 0

class ResolutionCamera:
    def __init__(self):
        self.low = 0
        self.medium = 1
        self.high = 2
        self.res_160x120 = 0
        self.res_320x240 = 1
        self.res_640x480 = 2

resolution = ResolutionCamera()

class Region:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

def Say(text, POST=True):
    global tts
    #print text
    try:
        #volume=GetTTSVolume()        
        #SetTTSVolume(0.99)
        if POST:
            tts.post.say(text)
        else:
            tts.say(text)
        #SetTTSVolume(volume)
    except NameError:
        print 'ALTextToSpeech proxy undefined. Are you running a simulated naoqi?'
        
def HeadTouch():
    head_touch = memoryProxy.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value", 0)
    return head_touch
#################################################################################
## Use this function, InitProxy, to initialise the proxy. As an argument give up
## the Ip of Nao
#################################################################################
def InitProxy(IP="marvin.local", proxy=[0], PORT = 9559):
    """proxy: (list) 1->TTS, 2->audio, 3->motion, 4->memory, 5->face, 6->video, 7->LED's, 8->Track, 9->Speech, 10->Audioplayer, 11->VisionToolbox"""
    global audioProxy
    global motionProxy
    global memoryProxy
    global cameraProxy
    global faceProxy
    global ledProxy
    global tts
    global trackfaceProxy
    global playProxy
    global videoProxy
    global speechProxy
    
    #printIP, PORT
    all_proxy = False

    proxy.sort()
    if proxy[0]==0:
        all_proxy = True
        proxy.pop(0)

    if all_proxy or proxy[0]==1:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            tts = naoqi.ALProxy("ALTextToSpeech", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALTextToSpeech established"
        except RuntimeError as e:
            print "Error when creating ALTextToSpeech proxy:"
            print str(e)

    if all_proxy or proxy[0]==2:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            audioProxy = naoqi.ALProxy("ALAudioDevice", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALAudioDevice established"
        except Exception, e:
            print "Error when creating TTS proxy:"
            print str(e)

    if all_proxy or proxy[0]==3:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:     
            motionProxy = naoqi.ALProxy("ALMotion", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALMotion established"
        except Exception, e:
            print "Error when creating motion proxy:"
            print str(e)

    if all_proxy or proxy[0]==4:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            memoryProxy = naoqi.ALProxy("ALMemory", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALMemory established"
        except Exception, e:
            print "Error when creating memory proxy:"
            print str(e)

    if all_proxy or proxy[0]==5:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            faceProxy = naoqi.ALProxy("ALFaceDetection", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALFaceDetection established"
        except Exception, e:
            print "Error when creating face detection proxy:"
            print str(e)

    if all_proxy or proxy[0]==6:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:    
            cameraProxy = naoqi.ALProxy("ALVideoDevice", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALVideoDevice established"
        except Exception, e:
            print "Error when creating video proxy:"
            print str(e)

    if all_proxy or proxy[0]==7:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:    
            ledProxy = naoqi.ALProxy("ALLeds", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALLeds established"
        except Exception, e:
            print "Error when creating ALLeds proxy:"
            print str(e)

    if all_proxy or proxy[0]==8:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:    
            trackfaceProxy = naoqi.ALProxy("ALFaceTracker", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALFaceTracker established"
        except Exception, e:
            print "Error when creating ALFaceTracker proxy:"
            print str(e)
            
    if all_proxy or proxy[0]==9:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:    
            speechProxy = naoqi.ALProxy("ALSpeechRecognition", IP, PORT)
	    sleep(0.1)
            #print "Proxy ALSpeechRecognition established"
        except Exception, e:
            print "Error when creating ALSpeechRecognition proxy:"
            print str(e)
            
    if all_proxy or proxy[0]==10:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            playProxy = naoqi.ALProxy("ALAudioPlayer", IP, 9559)
	    sleep(0.1)
            #print "Proxy connection is established"
        except Exception, e:
            print "Error when creating Audio Player proxy"
            print str(e)

    if all_proxy or proxy[0]==11:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            videoProxy = naoqi.ALProxy("ALVisionToolbox", IP, 9559)
	    sleep(0.1)
            #print "Proxy connection is established"
        except Exception, e:
            print "Error when creating Video recorder proxy"
            print str(e)
   
#################################################################################
## Use this function, CloseProxy, to close the proxy. As an argument give up
## the Ip of Nao
#################################################################################
def CloseProxy(proxy=[0]):
    """proxy: (list) 1->TTS, 2->audio, 3->motion, 4->memory, 5->face, 6->video, 7->LED's, 8->Track, 9->Speech, 10->Audioplayer, 11->VisionToolbox"""
    global audioProxy
    global motionProxy
    global memoryProxy
    global cameraProxy
    global faceProxy
    global ledProxy
    global tts
    global trackfaceProxy
    global playProxy
    global videoProxy
    global speechProxy
    
    #printIP, PORT
    all_proxy = False

    proxy.sort()
    if proxy[0]==0:
        all_proxy = True
        proxy.pop(0)

    if all_proxy or proxy[0]==1:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            tts.exit()
	    sleep(0.1)
            #print "Proxy ALTextToSpeech established"
        except RuntimeError as e:
            print "Error when deleting ALTextToSpeech proxy:"
            print str(e)

    if all_proxy or proxy[0]==2:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            audioProxy.exit()
	    sleep(0.1)
            #print "Proxy ALAudioDevice established"
        except Exception, e:
            print "Error when deleting TTS proxy:"
            print str(e)

    if all_proxy or proxy[0]==3:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:     
            motionProxy.exit()
	    sleep(0.1)
            #print "Proxy ALMotion established"
        except Exception, e:
            print "Error when deleting motion proxy:"
            print str(e)

    if all_proxy or proxy[0]==4:
        if len(proxy):

            proxy.pop(0)
            proxy.append(0)
        try:
            memoryProxy.exit()
	    sleep(0.1)
            #print "Proxy ALMemory established"
        except Exception, e:
            print "Error when deleting memory proxy:"
            print str(e)

    if all_proxy or proxy[0]==5:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            faceProxy.exit()
	    sleep(0.1)
            #print "Proxy ALFaceDetection established"
        except Exception, e:
            print "Error when deleting face detection proxy:"
            print str(e)

    if all_proxy or proxy[0]==6:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:    
            cameraProxy.exit()
	    sleep(0.1)
            #print "Proxy ALVideoDevice established"
        except Exception, e:
            print "Error when deleting video proxy:"
            print str(e)

    if all_proxy or proxy[0]==7:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:    
            ledProxy.exit()
	    sleep(0.1)
            #print "Proxy ALLeds established"
        except Exception, e:
            print "Error when deleting ALLeds proxy:"
            print str(e)

    if all_proxy or proxy[0]==8:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:    
            trackfaceProxy.exit()
	    sleep(0.1)
            #print "Proxy ALFaceTracker established"
        except Exception, e:
            print "Error when deleting ALFaceTracker proxy:"
            print str(e)
            
    if all_proxy or proxy[0]==9:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:    
            speechProxy.exit()
	    sleep(0.1)
            #print "Proxy ALSpeechRecognition established"
        except Exception, e:
            print "Error when deleting ALSpeechRecognition proxy:"
            print str(e)
            
    if all_proxy or proxy[0]==10:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            playProxy.exit()
	    sleep(0.1)
            #print "Proxy connection is established"
        except Exception, e:
            print "Error when deleting Audio Player proxy"
            print str(e)

    if all_proxy or proxy[0]==11:
        if len(proxy):
            proxy.pop(0)
            proxy.append(0)
        try:
            videoProxy.exit()
	    sleep(0.1)
            #print "Proxy connection is established"
        except Exception, e:
            print "Error when deleting Video recorder proxy"
            print str(e)
   



     
################################################################################
## nao.NaoFacePosition() subscribes to faceProxy and returns location of face.
## It uses the embedded functions of Aldebaran face detection. If you want to
## change the period, you will have to first unsubscribe using switch = False.
## It returns [face_location,detected]. detected = whether a face has been seen.
################################################################################
def ALFacePosition(switch = True, period = 100):    
    global alface_subscribed
    if alface_subscribed == False:
        faceProxy.subscribe("Test_Face", 500, 0.0)
        alface_subscribed = True
    location_face = memoryProxy.getData("FaceDetected")
    if switch == False:
        faceProxy.unsubscribe("Test_Face")
        alface_subscribed == False
    #print " location face: " , location_face
    if len(location_face) == 3:
        return [-location_face[1][0][0][1],location_face[1][0][0][2]], True
        
    else:
        return [], False

###############################################################################
## EyesLED() can change the color of the leds. The color parameter sets
## the color in RGB values.
## The standard color is off, [0,0,0]. The interpolation time defines the time
## in seconds it will take to fully switch to the new color.
###############################################################################
def EyeLED(color=[0,0,0],interpol_time = 0, POST=True):
    sGroup = "FaceLeds"
    try:
        if POST:
            ledProxy.post.fadeRGB(sGroup, 256*256*color[0] + 256*color[1] + color[2],interpol_time)
        else:
            ledProxy.fadeRGB(sGroup, 256*256*color[0] + 256*color[1] + color[2],interpol_time)            
                
    except NameError:
        print 'ALLeds proxy undefined.'

###############################################################################
## This function returns the available gestures located in the gesture dir.
###############################################################################
def GetAvailableGestures():
    """Returns available gestures in a list"""
    list_path = sys.path
    found = 0
    for i in range (0,len(list_path)):
        if os.path.exists(list_path[i]+"/gestures"):
            found = 1
            break

    if found == 0:
        print "Could not find /gestures directory!"
        raise IOError
        return None

    remove = []
    
    list_gestures = os.listdir(list_path[i]+"/gestures")
    for i in range(len(list_gestures)):
        list_gestures[i] = "/gestures/"+list_gestures[i]
        if not list_gestures[i].endswith(".py") and not list_gestures[i].endswith(".ges"):
            remove.append(i)

    ## remove non py files
    remove.reverse()
        
    for i in range(len(remove)):
        list_gestures.pop(remove[i])
        
    return list_gestures

###############################################################################
## This function returns the available gestures located in the gesture dir.
###############################################################################
def GetAvailableLEDPatterns():
    """Returns available gestures in a list"""
    list_path = sys.path
    found = 0
    for i in range (0,len(list_path)):
        if os.path.exists(list_path[i]+"/led"):
            found = 1
            break

    if found == 0:
        print "Could not find /led directory!"
        raise IOError
        return None

    list_led = os.listdir(list_path[i]+"/led")
    remove = []
    for i in range(len(list_led)):
        list_led[i] = "/led/"+list_led[i]
        if not list_led[i].endswith(".csv") and not list_led[i].endswith(".led"):
            remove.append(i)
    ## remove non csv files


    remove.reverse()
    for i in remove:
        list_led.pop(i)
            

    return list_led

###############################################################################
## This function returns the available dialogs located in the dialogs dir.
###############################################################################
def GetAvailableDialogs():
    """Returns available dialogs in a list"""
    list_path = sys.path
    found = 0
    for i in range (0,len(list_path)):
        if os.path.exists(list_path[i]+"/dialogs"):
            found = 1
            break

    if found == 0:
        print "Could not find /dialogs directory!"
        raise IOError
        return None

    list_dlg = os.listdir(list_path[i]+"/dialogs")
    remove = []
    for i in range(len(list_dlg)):
        list_dlg[i] = "/dialogs/"+list_dlg[i]
        if not list_dlg[i].endswith(".csv") and not list_dlg[i].endswith(".dlg"):
            remove.append(i)
    ## remove non csv files


    remove.reverse()
    for i in remove:

        list_dlg.pop(i)
            

    return list_dlg

#########################################################################
## Loads a dialog csv file and converts its logic and questions/messages
## to dictionaires for use in a smach state machine
#########################################################################
def LoadDialog(file_name):
    """ Give the filename of the dialog in the /dialogs folder. Extension should be .csv or .dlg."""
    list_path = sys.path
    filefound=False
    for i in range (0,len(list_path)):
        if os.path.exists(list_path[i]+"/dialogs/"+file_name):
    	    filefound=True
            break

    if not filefound:
        print "Dialog file "+str(file_name)+" not found in PYTHONPATH"
        return

    file_load = open(list_path[i]+"/dialogs/"+file_name)

    #read all rows of CSV file (assumes delimiter is ';')
    csv_reader = csv.reader(file_load, delimiter=';')

    return csv_reader

################################################################################
## nao.InitVideo() initialises the cv image and sets the variables on Nao.
## It allows you to give up the resolution. But first execute nao.InitProxy()
################################################################################
def InitVideo(resolution):
    global key
    global nameId
    global cameraProxy
    global cv_im

    resolutionar = [160,120],[320,240],[640,480]
    key=0
    random.random()*10
    try:
        nameId = cameraProxy.subscribe("python_GVM2"+str(random.random()*10), resolution, 0, 10) #0, 0, 10
    except NameError:
        print 'ALVideoDevice proxy undefined. Are you running a simulated naoqi?'
        return None
#    try:
#        cv_im = cv.CreateImageHeader((resolutionar[resolution][0],
#                                      resolutionar[resolution][1]),
#                                     cv.IPL_DEPTH_8U, 1)
#    except:
#        print "Cannot create image header"
#        return None
        
#################################################################################
## nao.GetImage() gets the image from Nao. You will fist need to execute
## nao.Initvideo()
#################################################################################
def GetImage():
    global img
    
    gotimage = False
    count = 0
    
    while not gotimage and count < 10:
        try:
            img =cameraProxy.getImageRemote(nameId)
#            pi=Image.frombuffer("L",(img[0],img[1]),img[6])
            gotimage =True
        except NameError:
            print 'ALVideoDevice proxy undefined. Are you running a simulated naoqi?'
            break
        except:
            count = count + 1
            print "problems with video buffer!! Did you initialize nao.InitVideo() the video first?"
#    cv.SetData(cv_im, pi.tostring())
#    cv.Flip(cv_im,cv_im,0)
#    key = cv.WaitKey(10)

#    return cv_im

################################################################################
## Initializes the track function it stiffens the joints, gathers the IDPose
################################################################################
def InitTrack():
    global xtargetold
    global ytargetold
    xtargetold = 0
    ytargetold = 0
    # Stiffening the head joints
    motionProxy.stiffnessInterpolation('HeadYaw', 1.0, 1.0)
    motionProxy.stiffnessInterpolation('HeadPitch', 1.0, 1.0)
    interpol_time = 0.5
    names  = ["HeadYaw","HeadPitch"]

################################################################################
## Releasing stiffness of the head joints
################################################################################
def EndTrack():
    motionProxy.stiffnessInterpolation('HeadYaw', 0.0, 1.0)
    motionProxy.stiffnessInterpolation('HeadPitch', 0.0, 1.0)

################################################################################
## If the tracking function is initialised you can let nao follow a point in
## the camera stream the boolean "detected" specifies whether the target
## was detected. "frametime" is the time between frames.
################################################################################   
def Track(target_loc, detected, speed = 5, min_move = 0.04):
    """
    target_loc =  the location Nao's head should move to in radians
    detected = is the head detected, If False target_loc is not used and speed of movement gradually decreases
    (optional) speed = the speed of the movement
    (optional) min_move = the minimal angle of difference between the target_loc and current location for movements to occur.

    """
    global xtargetold
    global ytargetold
    global time_old_track
    global id_pose
    global interpol_time
    global start_mov_t

    interpol_time = 1.0/speed

    xtarget = target_loc[0]
    ytarget = target_loc[1]

    try:
        frametime = time() - time_old_track
        time_old_track = time()
    except:
        print "Not able to determine frame rate. Guessing..."
        frametime = 0.15

    if detected == False:
        xtarget = xtargetold-xtargetold*(frametime)
        ytarget = ytargetold-ytargetold*(frametime)

    xtargetold = xtarget
    ytargetold = ytarget

    if ((xtarget > min_move or xtarget < -min_move) or (ytarget > min_move or ytarget < -min_move)):
        names  = ["HeadYaw","HeadPitch"]

        try:
            id_pose
        except NameError:
            id_pose = None

        if id_pose != None:
            motionProxy.stop(id_pose)

        try:
            id_pose = motionProxy.post.angleInterpolation(names, [-xtarget/2.5,ytarget/2.5] , interpol_time, False)
        except RuntimeError,e:
            print "Kan hoofd niet draaien"
        start_mov_t = time()
################################################################################



## Is used to see if Nao's head is moving.
################################################################################
def MovingHead():
    time_mov = time()-start_mov_t
    if time_mov > 2*interpol_time:
        return False
    else:
        return True
    return
###############################################################################
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


################################################################################
# Track Face
################################################################################
def ALTrack(switch=1):
    """Turn head tracking on or off. Or get status = 2"""
    if switch == 1:
        InitTrack()
        trackfaceProxy.startTracker()
    elif switch == 0:
        trackfaceProxy.stopTracker()
        EndTrack()
    else:
        return trackfaceProxy.isActive()
            

    
##############################################################################
## Put's Noa into its Initpose. Only use when standing or in crouch position.
#############################################################################
def InitPose(time_pos=0.5):
    """Nao will move to initpose."""
    dtheta = 0
    dx = 0
    dy = 0
    freq = 1
    motionProxy.setWalkTargetVelocity(dx, dy, dtheta, freq)
    sleep(0.1)
    # set stiffness
    motionProxy.stiffnessInterpolation('Body',1.0, time_pos)
    sleep(0.5)
    #IP.initPose(motionProxy)
    numJoints = len(motionProxy.getJointNames('Body'))

    allAngles = [0.0,0.0,                   # head
        1.39, 0.34, -1.39, -1.04, 0.0, 0.0,             # left arm
        0.0, 0.0, -0.43, 0.69, -0.34, 0.0,                  # left leg
        0.0, 0.0, -0.43, 0.69, -0.34, 0.0,                  # right leg
        1.39, -0.34, 1.39, 1.04, 0.0, 0.0]              # right arm
    #printnumJoints
    if (numJoints == 26):
        angles = allAngles
    elif (numJoints == 22):  # no hands (e.g. simulator)
        angles = allAngles[0:6] + allAngles[8:24]
    else:
        print "error in Init Pose"
        
    try:
        motionProxy.post.angleInterpolation('Body', angles, 1.5, True);
    
    except RuntimeError,e:
        print "An error has been caught"
        print e
        
def Stiffen(stiffness = True, int_time=1):
    """Make Nao stiffen its joints (Can be True or False)"""
    motionProxy.stiffnessInterpolation('Body',int(stiffness), int_time)

################################################################################
## Nao crouches and loosens it's joints.
###############################################################################
def Crouch():
    """Make Nao to crouch pose."""
    numJoints = len(motionProxy.getJointNames('Body'))
    
    allAngles = [0.0,0.0,                   # head
            1.545, 0.33, -1.57, -0.486, 0.0, 0.0,       # left arm
            -0.3, 0.057, -0.744, 2.192, -1.122, -0.035,     # left leg
            -0.3, 0.057, -0.744, 2.192, -1.122, -0.035,         # right leg
            1.545, -0.33, 1.57, 0.486, 0.0, 0.0]        # right arm
    
    if (numJoints == 26):
        angles = allAngles
    elif (numJoints == 22):  # no hands (e.g. simulator)
        angles = allAngles[0:6] + allAngles[8:24]
    else:
        print "error in numJoints"
            
    try:
        motionProxy.angleInterpolation('Body', angles, 1.5, True);

    except RuntimeError,e:
        print "An error has been caught"
        print e

    motionProxy.stiffnessInterpolation('Body',0, 0.5)
    
##################################################################################
## Allows Nao to move in a certain direction with a certain speed.
################################################################################
def Move(dx=0, dtheta=0, dy=0 ,freq=1):
    """"
    dx = forward speed, dtheta = rotational speed,
    dy = sidewards speed, freq = step frequency.
    Allows Nao to move in a certain direction
    with a certain speed.
    """
    
    motionProxy.setWalkTargetVelocity(dx, dy, dtheta, freq)

##################################################################################
## Allows Nao to move dx meters forward, dy meters sidways with final orientation of dtheta
################################################################################
def Walk(dx=0,dy=0,dtheta=0,post=False):
    
    
    """"
    dx = forward meters, dtheta = final angle,

    dy = sidewards meters
    Allows Nao to move in a certain direction.
    
    """
    if post==False:
        motionProxy.walkTo(dx, dy, dtheta)
    else:
        motionProxy.post.walkTo(dx, dy, dtheta)

##################################################################################
## Moves nao head yaw and pitch of the provided values yaw_val and pitch_val
################################################################################
def MoveHead(yaw_val=0, pitch_val=0, isAbsolute=True, post=True):

    names      = ["HeadYaw", "HeadPitch"]
    angleLists = [[yaw_val], [pitch_val]]
    timeLists  = [[1],[1]]
    if post==False:
        motionProxy.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    else:
        motionProxy.post.angleInterpolation(names, angleLists, timeLists, isAbsolute)

def GetYaw():
    
    

    names  = "HeadYaw"
    useSensors  = True
    HeadYaw = motionProxy.getAngles(names, useSensors)
    return HeadYaw

def GetPitch():
       
    names  = "HeadPitch"
    useSensors  = True
    HeadPitch = motionProxy.getAngles(names, useSensors)
    return HeadPitch


##################################################################################
## Allows Nao to play a sinusoidal wave of frequency in Hertz p1, Volume gain 0-100  p2, Stereo Pan set to either {-1,0,+1} p3 , duration in seconds
################################################################################

def PlaySine(p1,p2,p3,duration):
    global audioProxy
    
    try:
        audioProxy.playSine(p1,p2,p3,duration)
    except NameError:
        print 'ALAudioDevice proxy undefined. Are you running a simulated naoqi?'

###################
#stop music
###########################
def StopPlay():
    
    playProxy.stopAll()



######################################################
# Use this class and it's Play() function to play a wav or mp3 file on Nao.
# The files should be uploaded via ftp. Go to ftp://username:password@nao's_ip
# And upload them it's initial directory, which is /home/nao/ .
# id_music
######################################################



def Play(file_name):
    """Plays a audio file on Nao, it runs the file from the /home/nao/ directory"""
    file_name = "/home/nao/"+file_name
    id_music=playProxy.post.playFile(file_name)
    return id_music


###########
# Pause
########################
def Pause(id_music):

    playProxy.post.pause(id_music)


########################
#playFileFromPosition
##############################
def playFileFromPosition(file_name, position):
    file_name = "/home/nao/"+file_name
    id_music=playProxy.post.playFileFromPosition(file_name, position)
    return id_music


##########################
#Set Volume TTS
##################################
def SetTTSVolume(volume):
    tts.setVolume(volume)

##########################
#Get Volume TTS
##################################
def GetTTSVolume():
    vol=tts.getVolume()
    return vol

##########################
#Set Volume Music
##################################
def SetMusicVolume(id_music,volume):
    playProxy.setVolume(id_music,volume)

##########################
#Get Volume Music
##################################
def GetMusicVolume():
    vol=playProxy.getMasterVolume()
    return vol



###############################################################################
## This will record a file located in the /home/nao/naoqi/share/naoqi/vision
## Directory on Nao
###############################################################################
def Record(file_name, fps = 3.0, resolution = 0):
    """
    file_name without extension. fps, should be higher than 3.0.
    resolution shoud be between 0 and 2.
    Saved in /home/nao/naoqi/share/naoqi/vision
    """
    vidp.startVideoRecord_adv(file_name,fps,"MJPG",resolution,-1)

    

###############################################################################
## This function will look for a face. If it doesn't find it False is returned.
###############################################################################
def FindFace():
    """ It looks for a face and if it finds it returns boolean True """
    location, detected = ALFacePosition()
    if detected:
        return True
    
##     - -
##     0 0 0 0 0
##     4 2 0 2 4
##    ###########
##-0.4#+-+-+-+#e#
##    #|#####|#|#
##-0.2#+#+-+#+#+#
##    #|#|#|#|#|#
##   0#+#+#s#+#+#
##    #|#|###|#|#
## 0.2#+#+-+-+#+#
##    #|#######|#
## 0.4#+-+-+-+-+#
##    ###########
    
    offset=-0.2
#    offset=0.0
    gain=1.5
    yaw =   [0.0, 0.0,-0.2,-0.2,0.0,0.2, 0.2, 0.2, 0.0,-0.2,-0.4,-0.4,-0.4,-0.2,0.0,0.2,0.4, 0.4, 0.4, 0.2, 0.0]
    pitch = [0.0,-0.2,-0.2, 0.0,0.0,0.0,-0.2,-0.4,-0.4,-0.4,-0.4,-0.2, 0.0, 0.0,0.0,0.0,0.0,-0.2,-0.4,-0.4,-0.4]

#    pitch = [0,-0.2,-0.2,0.0,0.2,0.2,0.2,0.0,-0.2,-0.4,-0.4,-0.4,-0.4,-0.2,
#             0.0,0.2,0.4,0.4,0.4,0.4,0.4,0.2,0.0,-0.2,-0.4,0.0]

#    pitch = [0,-0.2,-0.2,0.0,0.0,0.0,0.2,0.0,-0.2,-0.4,-0.4,-0.4,-0.4,-0.2,
#             0.0,0.2,0.4,0.4,0.4,0.4,0.4,0.2,0.0,-0.2,0.0]

                                    
    for i in range(0,len(yaw)):
        names      = ["HeadYaw", "HeadPitch"]
        angleLists = [[yaw[i]*gain], [pitch[i]+offset]]
        timeLists  = [[1],[1]]
        isAbsolute = True
        motionProxy.angleInterpolation(names, angleLists, timeLists, isAbsolute)
        sleep(0.25)
        for i in range(0,4):
            sleep(0.25) #workaround from Marco -> try to improve face recognition
            location, detected = ALFacePosition()
            if detected:
                break
        if detected:
            return True
    return False
#######################################################################
## This functions executes movements transported from choregraph
## and saved in a *.py file. Make sure to initialize the motion proxy.
#######################################################################
def RunMovement(file_name, post = True, to_start_position = True):
    """ Give up the filename containing the movement. Needs motion proxy."""
    list_path = sys.path
    filefound = False
    for i in range (0,len(list_path)):
        if os.path.exists(list_path[i]+"/gestures/"+file_name):
	    file_name=list_path[i]+"/gestures/"+file_name
	    filefound=True
            break
        if os.path.exists(list_path[i]+"/"+file_name):
	    file_name=list_path[i]+"/"+file_name
	    filefound=True
            break

    if not filefound:
	print "Movement or gesture "+str(file_name)+" not found in PYTHONPATH"
	return

    file_load = open(file_name)
    lines = file_load.readlines()
    for i in range(0,len(lines)):
        if lines[i].startswith("try:"):
            break

        exec lines[i]

    if to_start_position:
        last_key = motionProxy.getAngles(names, True)

        high_time = 0.0
        for i in range(0,len(times)):
            cur_time = times[i][len(times[i])-1]
            if cur_time > high_time:
                high_time = cur_time

        for i in range(0, len(times)):
            times[i].append(high_time+0.1)
            times[i].append(high_time+2)
            keys[i].append(keys[i][len(keys[i])-1])
            keys[i].append([last_key[i],[ 3, -0.55556, 0.00000], [ 3, 0.55556, 0.00000]])

    
    if post:
        motionProxy.post.angleInterpolationBezier(names, times, keys)
    else:
        motionProxy.angleInterpolationBezier(names, times, keys)

###########################################################################
## This function runs a speech script saves as a *.csv file. Column 1
## contains the time in seconds, Column 2 contains the TTS input. This
## function requires a TTS proxy.
###########################################################################
def RunSpeech(file_name):
    """ file_name is the name containing the speech script."""
    list_path = sys.path
    file_found=False
    for i in range (0,len(list_path)):
        if os.path.exists(list_path[i]+"/tts/"+file_name):
	    file_name=list_path[i]+"/tts/"+file_name
	    filefound=True
            break
        if os.path.exists(list_path[i]+"/"+file_name):
	    file_name=list_path[i]+"/"+file_name
	    filefound=True
            break

    if not filefound:
	print "Speech file "+str(file_name)+" not found in PYTHONPATH"
	return

    try:
        script_reader = csv.reader(open(file_name, 'rb'))
    except:
        print "Speech script does not exist!!!"
        return
    cur_line = script_reader.next()
    start_time = time()
    while True:
        try:
            cur_line = script_reader.next()
        except:
            break
        while float(cur_line[0])> (time()-start_time):
            sleep(0.1)
        Say(cur_line[1])
        
########################################################################
## Uses a led CSV file to read out the proper eye pattern variables.
## Allows you to set LED Group, RGB, and Duration
## Frequency is currently ignored
## CSV file format:
##  Row 1 = Header (ignored)
##  Row 2-n = LED Group; Red; Green; Blue; Frequency; Duration
## Duration = Fade Time past to ALLeds.FadeListRGB
## CSV file delimiter is ';' (Used by Excel)
#########################################################################
def RunLED(file_name, post = True):
    """ Uses a led CSV file to read out the proper eye pattern variables."""
    #open CSV file
    list_path = sys.path
    filefound=False
    for i in range (0,len(list_path)):
        if os.path.exists(list_path[i]+"/led/"+file_name):
	    file_name=list_path[i]+"/led/"+file_name
	    filefound=True
            break
        if os.path.exists(list_path[i]+"/"+file_name):
	    file_name=list_path[i]+"/"+file_name
	    filefound=True
            break

    if not filefound:
	print "LED file "+str(file_name)+" not found in PYTHONPATH"
	return
     
    file_load = open(file_name, 'rb')

    #read all rows of CSV file (assumes delimiter is ';')
    csv_reader = csv.reader(file_load, delimiter=';')

    #read header row and ignore it
    csv_reader.next()

    #initialize LEDs to off
    ledProxy.post.off('FaceLeds')
    #print 'ledProxy.post.off(', 'FaceLeds', ')'

    #read first LED command and initialize fadeListRGB parameters
    parameters = csv_reader.next()
    name = parameters[0]
    rgbList = [256*256*int(parameters[1])+256*int(parameters[2])+int(parameters[3])]
    timeList = [float(parameters[5])]

    #while CSV file not empty
    while True:
         
         try:
              parameters = csv_reader.next()
         except:
              break

         #if LED Group different than last row
         if (name != parameters[0]):
               #send current fadeListRGB parameters to Nao
               ledProxy.post.fadeListRGB(name, rgbList, timeList)
               #print 'ledProxy.post.fadeListRGB(', name, rgbList, timeList, ')'
               #intialize fadeListRGB parameters
               name = parameters[0]
               rgbList = []
               timeList = []

         #add this row to fadeListRGB parameters
         rgbList.append(256*256*int(parameters[1])+256*int(parameters[2])+int(parameters[3]))
         timeList.append(float(parameters[5]))

    #all done - send current fadeListRGB parameters to Nao
    ledProxy.post.fadeListRGB(name, rgbList, timeList) 
    #print 'ledProxy.post.fadeListRGB(', name, rgbList, timeList, ')'
          
    return file_load

def GetAvailableModules():
    dir_file = []
    list_path = sys.path
    filefound = False
    for i in range (0,len(list_path)):
        if os.path.exists(list_path[i]+"/modules"):
            filefound = True
            break

    if not filefound:
        print "Could not find /modules directory!"
        raise IOError
        return None

    list_dir = os.listdir(list_path[i]+"/modules")

    for directory in list_dir:
        if not directory.startswith('.'):
            list_subdir = os.listdir(list_path[i]+"/modules/"+directory)
            module_files = ["/modules/",directory+"/"]
            for file_name in list_subdir:
                if not file_name.startswith("."):
                    module_files.append(file_name)
                    #module_files.append([directory,file_name])
            dir_file.append(module_files)
    return dir_file
    
def InitSpeech(wordList=["yes","no","hello NAO","goodbye NAO"]):
    global speechProxy
    global memoryProxy
    
#Creating a proxy on the module
#Before calling the ASR commands, you need to create a proxy on the ASR module:

#    asr = ALProxy("ALSpeechRecognition",myIP,9559) #IP = address of your robot
    asr=speechProxy
    
#Before starting the ASR engine, you must set the language of the speech recognition system. The list of the installed languages can be obtained through the getAvailableLanguages method.

    asr.setLanguage("English")

    # To set the words that should be recognized, use the setWordListAsVocabulary method.

#    print(wordList)
    asr.setWordListAsVocabulary(wordList)

    #Note:
    #The following feature (the usage of the "loadVocabulary()" function) is not available for Chinese and Japanese.
    #If you prefer not to use the setWordListAsVocabulary function, you can directly defined a vocabulary in a .lxd file and load it with the loadVocabulary method as described below:

    # Example: load the vocabulary defined in the file /home/nao/MyVocabulary.lxd asr.loadVocabulary(/home/nao/MyVocabulary.lxd)

    #Defining your vocabulary in a .lxd file and load it with the loadVocabulary function allows you to add alternative phonetic transcriptions that refine the automatic transcriptions of the ASR, and can improve performances. For example, if you want the speech recognition to be robust to the different pronunciations of the word "hello", you can define your vocabulary file as follows:

    #!Language=English #!FSG <words>=alt("yes" "no" "hello" "goodbye" )alt; <start>=alt(<words>)alt; #!Transcriptions hello h@l@U hello h@ll@U

    #The phonetic alphabet used to write these phonetizations is described here: http://www.phon.ucl.ac.uk/home/sampa/.

    #Collecting the recognized word in the memory
    #If a word has been recognized, the result is placed in the "WordRecognized" key of ALMemory.
    #As a result, you can read it by accessing this key in the ALMemory module.

    # Clear event WordRecognized in ALMemory module. 
    memoryProxy.insertData("WordRecognized",[])

    
def DetectSpeech():
    global memoryProxy
    try:
        #getData
        result=memoryProxy.getData("WordRecognized")
        if len(result)>1:
            memoryProxy.insertData("WordRecognized",[])

    except RuntimeError,e:
      # catch exception
      print "error getting data", e

    return result

if __name__ == "__main__":
    dir_file = GetAvailableModules()

    
   

    
    


