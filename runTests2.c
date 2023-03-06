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
#define curThetaLock 0
#define uLock 1
// This tests the multiproccessing enviroment

float curTheta;
float u = 0;
int desPower;
int dir;

PI_THREAD (IMU_thread){
    for(EVER){
    piLock(curThetaLock);
    curTheta = update_angle(1);
    piUnlock(curThetaLock);
    delay(10);
    }
}

PI_THREAD (motor_thread){
    for(EVER){

    piLock(uLock);
    desPower = u*1024/12;
    if(u<0){
        dir = 0; //Maybe the other way? Test and see 
    }
    else{
        dir = 1;
    }
    piUnlock(uLock);
    accuateMotor(desPower,dir,desPower,dir);
    delay(100);
    }
}

PI_THREAD (PID_thread){
    for(EVER){
    piLock(uLock);
    piLock(curThetaLock);
    //Add encoder speed as second argument to function
    //Add desired speed as third argument to function
    u =angleController(curTheta,0.0, 0.0);
    piUnlock(curThetaLock);
    piUnlock(uLock);
    delay(50);
    }
}

void setup(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering
    MPU6050_Init();
    
    initMotorPins(); //Initializes pins and hardware interupts for motors
    initRegParam(28.545755616, 0, 0, -0.0431, -0.0464);
    setupFirstValue();
}

int main(){
    setup();
    piThreadCreate (IMU_thread);
    piThreadCreate(PID_thread);
    piThreadCreate(motor_thread);
    for(EVER){
        if(abs(curTheta)>50){
        exit(1);
     }
     delay(100);
    }
}

