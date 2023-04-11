// Include standard libraries  
#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>
#include <math.h>
#include <MPU6050.h>
// Include project libraries 
extern "C" {
#include "motorControl.h"
}
extern "C" {
#include "pidMotor.h"
}
#include "Kalman.h"
#define EVER ;;
#define RAD_TO_DEG 57.2957795;

float curTheta;
float u = 0;
int desPower;
int dir;
float* speed;
MPU6050 imu(0x68);
Kalman kalman;
double kalTheta;
float gr, gp, gy;
float accX, accY, accZ;

// Sampling times
const float TIMU = 0.002;
const float Tmotor = 0.0025;
const float Tpid = 0.0025;

unsigned long curTime;
unsigned long lastIMUtime, lastmotorTime, lastpidTime;

void setup(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering

    imu.getAngle(0,&curTheta); //Calculate first value and input to filter 
    kalman.setAngle(curTheta); 
    

    initMotorPins(); //Initializes pins and hardware interupts for motors
    //setupFirstValue();
}

int main( int argc, char *argv[] ){

    if( argc != 6 ){
        printf("Wrong number of arguments, you had %d \n",argc);
        exit(2);
    }
    float Kp = (float)atof(argv[1]);
    float Ki = (float)atof(argv[2]);
    float Kd = (float)atof(argv[3]);
    float Kpv = (float)atof(argv[4]);
    float Kiv = (float)atof(argv[5]);
    initRegParam( Kp , Ki, Kd, Kpv, Kiv);

    setup();

    lastIMUtime = millis();
    lastmotorTime = millis(); 
    lastpidTime = millis();
    std::cout << "Starting up " << std::endl;
    for(EVER){

        curTime = millis();
        float dtIMU = (curTime-lastIMUtime)/1000.0f;
        if(dtIMU>=TIMU){
            //Update IMU

            /*
            imu.getAngle(0,&curTheta); //Uncomment to use complementary filter
            printf("Angle= %f \n",curTheta);
            */

            //Kalman filter
            
            imu.getGyro(&gr, &gp, &gy);
            imu.getAccel(&accX, &accY, &accZ);
            double roll  = atan(accY / sqrt(accX * accX + accZ * accZ)) * RAD_TO_DEG;
            curTheta = -kalman.getAngle(roll, gr, dtIMU);
            
            
            //std::cout << "CurTheta = "<< curTheta << std::endl;

            //Keep track of last time used
            lastIMUtime = curTime;
            if(dtIMU> TIMU*1.1){
                printf("Too slow time = %f \n",dtIMU);
            }
        }

        float dtPID = (curTime-lastpidTime)/1000.0f;
        if(dtPID>=Tpid){
            speed = calcSpeeds(0);
            //Calc u 
            u =angleController(curTheta,(speed[0]+speed[1])/2.0, 0.0,0);
            lastpidTime = curTime;
        }
        
        float dtMotor = (curTime-lastmotorTime)/1000.0f;
        if(dtMotor>=Tmotor){
            //desPower = fabs(u*1024.0/12.0)+100.0;
            desPower = fabs(u)+150; //Add 150 to account for startup torque 
            
            if(u<0){
                dir = 0; //Maybe the other way? Test and see 
            }
            else{
                dir = 1;
            }
            accuateMotor(desPower,dir,desPower,dir);
            lastmotorTime = curTime;
        }




        //Check for failure
        if(abs(curTheta)>50){
            accuateMotor(0,1,0,1);
            free(speed);
            exit(1);
        }
    }
}


