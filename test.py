import cv
import lib_headpose_NN


capture = cv.CaptureFromCAM(-1)
image = cv.QueryFrame(capture)
cv.ShowImage("image", image)
key = cv.WaitKey(30) # only works with cv window
image, center, detected, region = nao.Detect(image, False)
        
if detected:
    pitch, yaw = HeadPose(image, region)

