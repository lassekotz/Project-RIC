#include <stdio.h>
#include <cstdio>
#include <bcm2835.h>
#include "I2Cdev.h"
#include "MPU6050.h"
#include <math.h>




float IMU::getGyroRoll(int gyroX, int gyroBiasX, uint32_t lastTime)
{
  float gyroRoll;

  //integrate gyroscope value in order to get angle value
  gyroRoll = ((gyroX - gyroBiasX ) / 131) * ((float)(micros() - lastTime) / 1000000); 
  return gyroRoll;
}

