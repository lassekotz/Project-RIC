"""
    Script to collect data needed for ML
"""

import time
import IMU.IMU as imu
import smbus2					#import SMBus module of I2C
import math
import actuation.camera as cam
from time import sleep
import datetime
from threading import Thread


bus = smbus2.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address
imu.MPU_Init()
currAngle = imu.setupGyroTheta()
time1 = datetime.datetime.now()
#dt = 0.1 #Time step IMU
camera = cam.Camera()
camera.initialize()
i = 0
"""
while(True):
    t2 = datetime.datetime.now()
    dt = (t2 - t1)
    currAngle = imu.Update_angle(currAngle,dt.total_seconds())
    
    #
    
    print(currAngle)
    t1 = t2
    sleep(0.1)
    i += 1
   """ 

def runA():
    while True:
            time2 = datetime.datetime.now()
            dt = (time2 - time1)
            currAngle = imu.Update_angle(currAngle,dt.total_seconds())
    
            print(currAngle)
            time1 = time2
            sleep(0.1)

def runB():
    while True:
        camera.capture_image(i , resolution = (960, 540))
        i +=1

if __name__ == "__main__":
    t1 = Thread(target = runA)
    t2 = Thread(target = runB)
    t1.setDaemon(True)
    t2.setDaemon(True)
    t1.start()
    t2.start()
    while True:
        pass
