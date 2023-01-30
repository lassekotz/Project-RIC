import time
import IMU.IMU as imu
import smbus2					#import SMBus module of I2C
import math
import actuation.camera as cam
from time import sleep
import datetime
from multiprocessing import Process

def runA():
    bus = smbus2.SMBus(1)  # or bus = smbus.SMBus(0) for older version boards
    Device_Address = 0x68  # MPU6050 device address
    imu.MPU_Init()
    currAngle = imu.setupGyroTheta()
    time1 = datetime.datetime.now()
    while True:
        time2 = datetime.datetime.now()
        dt = (time2 - time1)
        currAngle = imu.Update_angle(currAngle, dt.total_seconds())

        #print(currAngle)
        time1 = time2
        sleep(0.001)

        return currAngle


def runB():
    pass



def reset_pendulum(threshold):
    imu.MPU_Init()
    currAngle = imu.setupGyroTheta()
    time1 = datetime.datetime.now()

    t1 = Process(target=runA)
    #t2 = Process(target=runB)

    #while (abs(currAngle) > threshold):
    while True:
        time2 = datetime.datetime.now()
        dt = (time2 - time1)
        currAngle = imu.Update_angle(currAngle, dt.total_seconds())

        sleep(0.001)

        #return currAngle
        
        if (abs(currAngle) > threshold):
            print("False")
        else:
            print("True")

reset_pendulum(10)