import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import importlib
import time as time
from BaseMotor import Motor
import sys
from picamera import PiCamera
from camera import Camera

def main(motor, ang_lim, camera):#, Camera):
    IMU_datalist = []
    #camera.initialize()
    j = 1
    for i in range(round(ang_lim / 0.18)): #.18 is stepsize in degrees
        motor.run()
        ext = '.jpg'
        imname = 'picture%s%s' %(j, ext)
        print(imname)
        camera.capture(imname)
        
        time.sleep(1)
        j += 1
        
        
        
        #TODO PRINT IMU-DATA
        #TODO CAPTURE IMAGE
        
        #IMU_datalist.append(imu_datapoint)


motor = Motor([24,25,8,7])
camera = PiCamera()


main(motor, 10, camera)
