'''
		Read Gyro and Accelerometer by Interfacing Raspberry Pi with MPU6050 using Python
	http://www.electronicwings.com
'''
import smbus					#import SMBus module of I2C
from time import sleep          #import
import math

#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47
thetaOld = 0


def MPU_Init():
	#write to sample rate register
	bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
	
	#Write to power management register
	bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
	
	#Write to Configuration register
	bus.write_byte_data(Device_Address, CONFIG, 0)
	
	#Write to Gyro configuration register
	bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
	
	#Write to interrupt enable register
	bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
	#Accelero and Gyro value are 16-bit
		high = bus.read_byte_data(Device_Address, addr)
		low = bus.read_byte_data(Device_Address, addr+1)
	
		#concatenate higher and lower value
		value = ((high << 8) | low)
		
		#to get signed value from mpu6050
		if(value > 32768):
				value = value - 65536
		return value





def Update_angle(thetaOld,dt):

	#Read Accelerometer raw value
	acc_x = read_raw_data(ACCEL_XOUT_H)
	acc_y = read_raw_data(ACCEL_YOUT_H)
	acc_z = read_raw_data(ACCEL_ZOUT_H)
	
	#Read Gyroscope raw value
	gyro_x = read_raw_data(GYRO_XOUT_H)
	gyro_y = read_raw_data(GYRO_YOUT_H)
	gyro_z = read_raw_data(GYRO_ZOUT_H)
	
	#Full scale range +/- 250 degree/C as per sensitivity scale factor
	AccX = acc_x/16384.0
	AccX = acc_y/16384.0
	AccZ = acc_z/16384.0
	
	GyroX = gyro_x/131.0
	GyroY = gyro_y/131.0
	GyroZ = gyro_z/131.0


	thetaA = 180/3.1415* math.atan2(-AccX,math.sqrt(math.pow(AccY,2)+ math.pow(AccZ,2))) #Acceleromter angle 

	thetaG = thetaOld + GyroY * dt # Gyro angle [deg/s * s = deg]

	theta = 0.95 * thetaG + 0.05 * thetaA #Complementary filter 
	return theta

def setupGyroTheta(): #Initialises gyro value as accelerometer value to speed up filter 
	#Read Accelerometer raw value
	acc_x = read_raw_data(ACCEL_XOUT_H)
	acc_y = read_raw_data(ACCEL_YOUT_H)
	acc_z = read_raw_data(ACCEL_ZOUT_H)

	#Full scale range +/- 250 degree/C as per sensitivity scale factor
	AccX = acc_x/16384.0
	AccX = acc_y/16384.0
	AccZ = acc_z/16384.0

	theta = 180/3.1415* math.atan2(-AccX,math.sqrt(math.pow(AccY,2)+ math.pow(AccZ,2))) #Acceleromter angle 
	return theta


currAngle = setupGyroTheta()
while(1):
	currAngle = Update_angle(currAngle)
	sleep(0.1)
	print("Current angle: "+currAngle)
