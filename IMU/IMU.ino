/* 
  MPU6050 IMU angle inputs
*/
#include <Wire.h>
const int MPU = 0x68; //I2C address 
float AccX, AccY, AccZ;
float GyroX, GyroY, GyroZ; 
float accAngleX, accAngleY, gyroAngleX, gyroAngleY, gyroAngleZ;
float roll, pitch, yaw;
float elapsedTime, currentTime, previousTime;
float theta,thetaA,thetaG;

void setup() {
  
  Serial.begin(19200);
  Wire.begin();
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);                  // Talk to the register 6B
  Wire.write(0x00);                  // Make reset - place a 0 into the 6B register
  Wire.endTransmission(true);

  delay(50);
  thetaG = setupGyroTheta();
}

void loop() {
  
  //Accelerometer data
  Wire.beginTransmission(MPU);
  Wire.write(0x3B); //Register 0x3B (Accelerometer)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 6, true); // Read 6 registers total, each axis value is stored in 2 registers
  //For a range of +-2g, we need to divide the raw values by 16384, according to the datasheet
  AccX = (Wire.read() << 8 | Wire.read()) / 16384.0; // X-axis value
  AccY = (Wire.read() << 8 | Wire.read()) / 16384.0; // Y-axis value
  AccZ = (Wire.read() << 8 | Wire.read()) / 16384.0; // Z-axis value

  //Serial.print("Acc theta: ");
  thetaA = 180/3.1415*atan2(-AccX,sqrt(pow(AccY,2)+ pow(AccZ,2) ));
  //Serial.println(thetaA);
  

  //Gyro data 
  previousTime = currentTime;        // Previous time is stored before the actual time read
  currentTime = millis();            // Current time actual time read
  elapsedTime = (currentTime - previousTime) / 1000; // Divide by 1000 to get seconds
  Wire.beginTransmission(MPU);
  Wire.write(0x43); // Gyro data first register address 0x43
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 6, true); // Read 4 registers total, each axis value is stored in 2 registers
  GyroX = (Wire.read() << 8 | Wire.read()) / 131.0; // For a 250deg/s range we have to divide first the raw value by 131.0, according to the datasheet
  GyroY = (Wire.read() << 8 | Wire.read()) / 131.0;
  GyroZ = (Wire.read() << 8 | Wire.read()) / 131.0;
  // Currently the raw values are in degrees per seconds, deg/s, so we need to multiply by sendonds (s) to get the angle in degrees
  thetaG = theta + GyroY * elapsedTime; // deg/s * s = deg
  gyroAngleY = gyroAngleY + GyroY * elapsedTime;
  yaw =  yaw + GyroZ * elapsedTime;
  //Serial.print("Gyro theta: ");
  //Serial.println(thetaG);

  
  // Complementary filter - combine acceleromter and gyro angle values
  roll = 0.96 * gyroAngleX + 0.04 * accAngleX;
  pitch = 0.96 * gyroAngleY + 0.04 * accAngleY;

  theta = 0.95 * thetaG + 0.05 * thetaA;
  //Serial.print("Fused theta: ");
  Serial.println(theta);

  delay(100);
}

float setupGyroTheta() {
  //Accelerometer data
  Wire.beginTransmission(MPU);
  Wire.write(0x3B); //Register 0x3B (Accelerometer)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 6, true); // Read 6 registers total, each axis value is stored in 2 registers
  //For a range of +-2g, we need to divide the raw values by 16384, according to the datasheet
  AccX = (Wire.read() << 8 | Wire.read()) / 16384.0; // X-axis value
  AccY = (Wire.read() << 8 | Wire.read()) / 16384.0; // Y-axis value
  AccZ = (Wire.read() << 8 | Wire.read()) / 16384.0; // Z-axis value

  Serial.print("Acc theta: ");
  float initThetaG = 180/3.1415*atan2(-AccX,sqrt(pow(AccY,2)+ pow(AccZ,2) ));
  Serial.println(thetaA);
  return initThetaG;
}


