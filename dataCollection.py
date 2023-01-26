"""
    Script to collect data needed for ML
"""

import time
import IMU.IMU as imu
import smbus2					#import SMBus module of I2C
import math
#import actuation.dataCollection as actu
import actuation.camera as cam
from time import sleep
import datetime




bus = smbus2.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address
imu.MPU_Init()
currAngle = imu.setupGyroTheta()
t1 = datetime.datetime.now()
#dt = 0.1 #Time step IMU


while(True):
    t2 = datetime.datetime.now()
    dt = (t2 - t1)
    currAngle = imu.Update_angle(currAngle,dt.total_seconds())
    
    print(currAngle)
    t1 = t2
    sleep(0.001)
    

