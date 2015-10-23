# After initialisation self.Orientation can be used to change the orientation of
# the head
from panda3d.core import loadPrcFileData
loadPrcFileData('', 'win-origin 650 340')
loadPrcFileData('', 'win-size 640 480')

import direct.directbase.DirectStart
from pandac.PandaModules import DirectionalLight
from pandac.PandaModules import Vec4
#from direct.showbase.ShowBase import ShowBase



class Head3D:
  def __init__(self):
    self.yaw = 0
    self.pitch = 0
    self.roll = 0

    #load headsmall.egg 3d Model
    self.model = loader.loadModel("head.bam")

    # Put her in the scene.
    self.model.reparentTo(render)

    # Location Model
    self.model.setPosHpr(0,0,0,0,0,0)

    # Position Camera
    base.trackball.node().setPos(0, 10, 0)
    self.model.clearColor()

    # Change texture color
    self.model.setColorScale(0.90, 0.55, 0.40, 1.0)

    # Directional light
    self.directionalLight = DirectionalLight('directionalLight')
    self.directionalLight.setColor(Vec4(1, 1, 1, 1))
    self.directionalLightNP = render.attachNewNode(self.directionalLight)
    # This light is facing forwards, away from the camera.
    self.directionalLightNP.setHpr(0, -20, 0)
    render.setLight(self.directionalLightNP)

  def Orientation(self, yaw, pitch, roll = 0):
    self.yaw = yaw
    self.pitch = pitch
    self.roll = roll
    taskMgr.add(self.AddOrientationTask, 'ChangeOrientation')
    taskMgr.step()
    taskMgr.step()

  def AddOrientationTask(self, task):
    self.model.setPosHpr(0,0,0,self.yaw,self.pitch,self.roll)
    return task.done


