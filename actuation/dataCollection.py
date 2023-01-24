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

def main(motor1, motor2, ang_lim, camera):
    j = 1
<<<<<<< HEAD
    imlist = []
    nr_of_steps = 3
    for i in range(round(ang_lim / (nr_of_steps*0.18))): #.18 is stepsize in degrees
        motor1.run(True)
        #motor2.run()
        #ext = '.jpg'
        #imgname = 'Images/picture%s%s' %(j, ext) #format image name string
        #print(imgname)
        #camera.capture(imgname)
=======
    for i in range(round(ang_lim / 0.18)): #.18 is stepsize in degrees
        motor1.run()
        motor2.run()
        ext = '.jpg'
        imgname = 'Images/picture%s%s' %(j, ext) #format image name string
        print(imgname)
        camera.capture(imgname)
>>>>>>> be4848d18d33f5f144b9c1eb734f841d9d691b98
        
        time.sleep(1)
        j += 1

        #TODO PRINT IMU-DATA
        #TODO CAPTURE IMAGE
        
        #IMU_datalist.append(imu_datapoint)

<<<<<<< HEAD

motor1 = Motor([14, 15, 18, 23])
#motor2 = Motor([24, 25, 8, 7])
camera = PiCamera()


anglim = float(sys.argv[1])
main(motor1, anglim, camera)
motor1.shutdown()
=======
motor1 = Motor([14, 15, 18, 23])
motor2 = Motor([24,25,8,7])
camera = PiCamera()

anglim = float(sys.argv[1])
main(motor1, motor2, anglim, camera)

>>>>>>> be4848d18d33f5f144b9c1eb734f841d9d691b98
