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
from multiprocessing import Process, Value
import serial
ser = serial.Serial('/dev/ttyACM0',9600, timeout =.1)

def writeToSerial(x): #Input a simple string with a number i.e "20"
    ser.write( (x).encode())
    time.sleep(0.05)
    return 0


def runA():
    bus = smbus2.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
    Device_Address = 0x68   # MPU6050 device address
    imu.MPU_Init()
    global currAngle
    currAngle = imu.setupGyroTheta()
    time1 = datetime.datetime.now()
    while True:
            time2 = datetime.datetime.now()
            dt = (time2 - time1)
            currAngle = imu.Update_angle(currAngle,dt.total_seconds())
    
            print(currAngle)
            time1 = time2
            sleep(0.001)

def runB():
    camera = cam.Camera()
    camera.initialize()
    i = 0
    while True:
        camera.capture_image(i , resolution = (960, 540))
        i +=1
        #writeToSerial("INSERT STEPS TO SEND HERE")
        sleep(1)

def setupPendulum(angle, threshold):
    while (abs(angle) > threshold):
        if (angle < 0):
            writeToSerial(1)
        else:
            writeToSerial(-1)
            








if __name__ == "__main__":
    currAngle = Value('d', 0.0)
    t1 = Process(target = runA)
    t2 = Process(target = runB)
    t1.start()
    setupPendulum(currAngle, 10)
    t1.join()
    t2.start()
    while True:
        pass
