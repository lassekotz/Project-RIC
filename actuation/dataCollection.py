import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import importlib
import time as time
from BaseMotor import Motor
import sys
from picamera import PiCamera
from camera import Camera
from driveUpload import drive_upload
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import sys

def main(motor1, ang_lim, camera):
    IMU_datalist = []
    j = 1
    imlist = []
    for i in range(round(ang_lim / 0.18)): #.18 is stepsize in degrees
        motor1.run(False)
        #motor2.run(True)
        ext = '.jpg'
        imgname = 'Images/picture%s%s' %(j, ext) #format image name string
        print(imgname)
        #camera.capture("/Images", imgname)
        camera.capture(imgname)
        
        time.sleep(1)
        j += 1

        #TODO PRINT IMU-DATA
        #TODO CAPTURE IMAGE
        
        #IMU_datalist.append(imu_datapoint)


motor1 = Motor([24,25,8,7])
#motor2 = Motor([])
camera = PiCamera()


#imlist = main(motor1, motor2, 10, camera)
anglim = sys.argv[1]
main(motor1, anglim, camera)

drive_upload()