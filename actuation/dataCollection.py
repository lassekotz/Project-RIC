#import RPi.GPIO as GPIO
#from RpiMotorLib import RpiMotorLib
import importlib
import time as time
from BaseMotor import Motor
import sys
from BaseMotor import Motor

#camera = BaseCamera()

def main(Motor, ang_lim, Camera):
    IMU_datalist = []
    Camera.initialize()
    for i in range(round(ang_lim / 0.18)): #.18 is stepsize in degrees
        run()
        Camera.capture_image()
        time.sleep(.5)

        #TODO PRINT IMU-DATA
        #TODO CAPTURE IMAGE
        
        #IMU_datalist.append(imu_datapoint)
        #


motor = Motor([24,25,8,7])
motor.shutdown()