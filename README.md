# Head pose estimation

## Start program:
From the prompt:

python lib_headpose_NN.py (PT2 released version)

or 

python lib_headpose_NN2.py (variation)


From window:

Doubleclick the file lib_headpose_NN.py

or

Doubleclick the file lib_headpose_NN2.py

## Stop program:
Press the Esc key when the main window showing the webcam image is active.
Usage:
	import lib_headpose_NN
	pitch, yaw = HeadPose(image, region)

## Examples
### Example using webcam:
	import lib_headpose_NN
	capture = cv.CaptureFromCAM(-1)
	image = cv.QueryFrame(capture)
    cv.ShowImage("image", image)
    key = cv.WaitKey(30) # only works with cv window
    image, center, detected, region = nao.Detect(image, False)
        
    if detected:
		pitch, yaw = HeadPose(image, region)

### Example using nao cam
	import lib_headpose_NN
	nao.InitProxy(‘your Nao IP’
	nao.InitVideo()
	image=nao.GetImage()
    cv.ShowImage("image", image)
    key = cv.WaitKey(30) # only works with cv window
    image, center, detected, region = nao.Detect(image, False)
        
    if detected:
		pitch, yaw = HeadPose(image, region)