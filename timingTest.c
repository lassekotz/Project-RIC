// Include standard libraries  
#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>
#include <math.h>
// Include project libraries 
#include "motorControl.h"
#include "pidMotor.h"
#include "call_IMU.h"
#define EVER ;;

float curTheta;
float u = 0;
int desPower;
int dir;
float* speed;

// Sampling times
const float TIMU = 0.01;
const float Tmotor = 0.1;
const float Tpid = 0.08;

unsigned long curTime;
unsigned long lastIMUtime, lastmotorTime, lastpidTime;

void setup(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering
    MPU6050_Init();
    
    initMotorPins(); //Initializes pins and hardware interupts for motors
    initRegParam(47.2980677808111, 28.1697176964395, 17.4085518777762,8538.19944333327,  -0.1047, -0.087568,Tpid);
    setupFirstValue();
}

int main(){
    setup();

    lastIMUtime = millis();
    lastmotorTime = millis(); 
    lastpidTime = millis();

    for(EVER){

        curTime = millis();
        float dtIMU = (curTime-lastIMUtime)/1000.0f;
        if(dtIMU>=TIMU){
            //Update IMU
            curTheta = update_angle(1);
            lastIMUtime = curTime;
        }

        float dtPID = (curTime-lastpidTime)/1000.0f;
        if(dtPID>=Tpid){
            speed = calcSpeeds(1);
            //Calc u 
            u =angleController(curTheta,0.0, 0.0);
            lastpidTime = curTime;
        }
        
        float dtMotor = (curTime-lastmotorTime)/1000.0f;
        if(dtMotor>=Tmotor){
            desPower = fabs(u*1024.0/12.0);
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
        if(abs(curTheta)>15){
            accuateMotor(0,1,0,1);
            exit(1);
        }
    }
}


