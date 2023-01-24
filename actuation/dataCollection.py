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
    nr_of_steps = 3
    for i in range(round(ang_lim / (nr_of_steps*0.18))): #.18 is stepsize in degrees
        motor1.run(True)
        #motor2.run()
        #ext = '.jpg'
        #imgname = 'Images/picture%s%s' %(j, ext) #format image name string
        #print(imgname)
        #camera.capture(imgname)
        
        time.sleep(1)
        j += 1

        #TODO PRINT IMU-DATA
        #TODO CAPTURE IMAGE
        
        #IMU_datalist.append(imu_datapoint)


motor1 = Motor([14, 15, 18, 23])
#motor2 = Motor([24, 25, 8, 7])
camera = PiCamera()


anglim = float(sys.argv[1])
main(motor1, anglim, camera)
motor1.shutdown()
