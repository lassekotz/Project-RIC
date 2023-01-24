"""
    Script to collect data needed for ML
"""

import sched
import time
import IMU.IMU as imu
import smbus2					#import SMBus module of I2C
import math
import actuation.dataCollection as actu



bus = smbus2.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address
currAngle = imu.setupGyroTheta()
dt = 0.1 #Time step IMU

scheduler = sched.scheduler(time.time,
                            time.sleep)


imu_event = scheduler.enter(dt,1,
    currAngle=imu.Update_angle(currAngle,dt))


#Insert shitty Lasse event functions here 

scheduler.run()
