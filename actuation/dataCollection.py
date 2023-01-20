import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import time as time
from BaseMotor import Motor
from camera import Camera

    def main(Motor.run, ang_lim, Camera):
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