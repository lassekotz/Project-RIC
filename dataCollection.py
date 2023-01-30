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
from multiprocessing import Process, Value, Manager, Pool, SimpleQueue
import serial
import random as rd


ser = serial.Serial('/dev/ttyACM0',9600, timeout =.1)

def writeToSerial(x): #Input a simple string with a number i.e "20"
    ser.write( (x).encode())
    time.sleep(0.05)
    return 0


def runA(currAngle):
    bus = smbus2.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
    Device_Address = 0x68   # MPU6050 device address
    imu.MPU_Init()
    
    #global currAngle
    #currAngle = imu.setupGyroTheta()
    time1 = datetime.datetime.now()
    while True:
        #print(currAngle)
        time2 = datetime.datetime.now()
        dt = (time2 - time1)
        currAngle = imu.Update_angle(currAngle,dt.total_seconds())
    
        time1 = time2
        currAngleG.value = currAngle
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

def setupPendulum(currAngle, threshold):
    while (abs(currAngle) > threshold):
        #currAngle = imu.setupGyroTheta()
        #print(currAngleG.value)
        if (currAngleG.value <= 0):
            writeToSerial(str(-10))
        else:
            writeToSerial(str(10))
            
        time.sleep(1)
        
def gaussianMovement():
    anglist = rd.gauss(0, 5)
    

if __name__ == "__main__":
    threshold = Value('d', 10.0)
    currAngleG = Value('d', imu.setupGyroTheta())
        
    t1 = Process(target = runA, args = (currAngleG.value, ))
    t3 = Process(target = setupPendulum, args = (currAngleG.value, threshold.value))
    t4 = 
    
    t1.start()
    t3.start()
    t3.terminate()
    t4.start()
    
    
    
    #t1.join()
    #t3.join()

    #t2.start()
    while True:
        pass


