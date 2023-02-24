// Get angles from MPU 6050 data with Raspberry pi

#include <wiringPiI2C.h>
#include <stdlib.h>
#include <stdio.h>
#include <wiringPi.h>
#include <math.h>
#include "call_IMU.h"

#define MPU 0x68	/*Device Address/Identifier for MPU6050*/

#define PWR_MGMT_1   0x6B
#define SMPLRT_DIV   0x19
#define CONFIG       0x1A
#define GYRO_CONFIG  0x1B
#define INT_ENABLE   0x38
#define ACCEL_XOUT_H 0x3B
#define ACCEL_YOUT_H 0x3D
#define ACCEL_ZOUT_H 0x3F
#define GYRO_XOUT_H  0x43
#define GYRO_YOUT_H  0x45
#define GYRO_ZOUT_H  0x47


void MPU6050_Init(){
	fd = wiringPiI2CSetup(0x68);   /*Initializes I2C with device Address*/
	wiringPiI2CWriteReg8 (fd, SMPLRT_DIV, 0x07);	/* Write to sample rate register */
	wiringPiI2CWriteReg8 (fd, PWR_MGMT_1, 0x01);	/* Write to power management register */
	wiringPiI2CWriteReg8 (fd, CONFIG, 0);		/* Write to Configuration register */
	wiringPiI2CWriteReg8 (fd, GYRO_CONFIG, 24);	/* Write to Gyro Configuration register */
	wiringPiI2CWriteReg8 (fd, INT_ENABLE, 0x01);	/*Write to interrupt enable register */

	} 
short read_raw_data(int addr){
	short high_byte,low_byte,value;
	high_byte = wiringPiI2CReadReg8(fd, addr);
	low_byte = wiringPiI2CReadReg8(fd, addr+1);
	value = (high_byte << 8) | low_byte;
	return value;
}

void ms_delay(int val){
	int i,j;
	for(i=0;i<=val;i++)
		for(j=0;j<1200;j++);
}



double update_angle(int verbose){
	
		//Keep track of time
   		u_int curT = millis();
   		double dt = (double)(curT-oldT);
	
		/*Read raw value of Accelerometer and gyroscope from MPU6050*/
		Acc_x = read_raw_data(ACCEL_XOUT_H);
		Acc_y = read_raw_data(ACCEL_YOUT_H);
		Acc_z = read_raw_data(ACCEL_ZOUT_H);
		
		Gyro_x = read_raw_data(GYRO_XOUT_H);
		Gyro_y = read_raw_data(GYRO_YOUT_H);
		Gyro_z = read_raw_data(GYRO_ZOUT_H);
		
		/* Divide raw value by sensitivity scale factor */
		Ax = Acc_x/16384.0;
		Ay = Acc_y/16384.0;
		Az = Acc_z/16384.0;
		
		Gx = Gyro_x/131.0;
		Gy = Gyro_y/131.0;
		Gz = Gyro_z/131.0;

		// Acc angle
		thetaA = 180.0/3.1415* atan2(-Ax,sqrt(pow(Ay,2)+ pow(Az,2)));

		// Gyro angle
		thetaG = thetaOld + GyroY * dt; // deg/s * s = deg

		// Complementary filter
		theta = 0.95 * thetaG + 0.05 * thetaA;

		// Store for next loop
   		oldT = curT;
		
		if(verbose){
			printf("\n Theta=%.3d °\tThetaG=%.3d °\tThetaA=%.3d °\n",theta,thetaG,thetaA);
		}
		thetaOld = theta;
		return theta;
	
		
}

void setupFirstValue(){
	//Initializes first gyro value from accelerometer to speed up start
	Acc_x = read_raw_data(ACCEL_XOUT_H);
	Acc_y = read_raw_data(ACCEL_YOUT_H);
	Acc_z = read_raw_data(ACCEL_ZOUT_H); 
	Ax = Acc_x/16384.0;
	Ay = Acc_y/16384.0;
	Az = Acc_z/16384.0;

	thetaG = 180.0/3.1415* atan2(-AccX,sqrt(pow(AccY,2)+ pow(AccZ,2)));
}




//gcc IMU.c -o IMU -l wiringPi -lm