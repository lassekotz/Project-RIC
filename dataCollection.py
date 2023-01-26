"""
    Script to collect data needed for ML
"""

import time
import ctypes
import smbus2					#import SMBus module of I2C
import math
import actuation.dataCollection as actu
import actuation.camera as cam
from time import sleep

imufunc = ctypes.CDLL("/IMU/build/libimu.so")



bus = smbus2.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address
currAngle = imufunc.setupGyroTheta()
dt = 0.1 #Time step IMU
imufunc.update_angle.argtypes = [ctypes.c_double]

while(True):
    currAngle = imufunc.update_angle(currAngle,dt)
    print(currAngle)
    sleep(0.1)
    
