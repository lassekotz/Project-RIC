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


ser = serial.Serial('/dev/ttyACM0',9600, timeout =.1, rtscts = True)

def writeToSerial(x): #Input a simple string with a number i.e "20"
    ser.write( (x).encode())
    time.sleep(0.05)
    return 0


def runA(currAngle):
    bus = smbus2.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
    Device_Address = 0x68   # MPU6050 device address
    imu.MPU_Init()

    time1 = datetime.datetime.now()
    while True:
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

def movePendulum(threshold, nr_of_samples):
    step_tick_size = 5
    mu = 0
    std = 5
    
    print("Calibrating pendulum to invert... please stand by")
    while (abs(currAngleG.value) > threshold):

        if (currAngleG.value <= 0):
            writeToSerial(str(step_tick_size))
        else:
            writeToSerial(str(-step_tick_size))
        time.sleep(1)
    print("Finished calibration")
    
    i = 0 #nr of samples
    while (i < nr_of_samples):
        time.sleep(3)
        nextPos = rd.gauss(mu, std)
        while (abs(nextPos) > 85):
            nextPos = rd.gauss(0,5)
        angDist = nextPos - currAngleG.value
        nr_of_steps = angDist/(360/2048)
        nr_of_ticks = nr_of_steps/step_tick_size

        j = 0 #nr of tick
        print("=============")
        while (j < abs(nr_of_steps)/step_tick_size):
            print("Sample " + str(i + 1) + "   |   " + "tick: " + str(j) + "/" + str(round(abs(nr_of_steps)/step_tick_size)))
            writeToSerial(str(round(nr_of_ticks)))
            time.sleep(1)
            j += 1
        i += 1
            
if __name__ == "__main__":
    threshold = Value('d', 1.0)
    currAngleG = Value('d', imu.setupGyroTheta())
        
    t1 = Process(target = runA, args = (currAngleG.value, ))
    t3 = Process(target = movePendulum, args = (threshold.value, 20))
    
    t1.start()
    t3.start()


    
    #t2.start()
    while True:
        pass

