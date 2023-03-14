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
double speed;


void setup(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering
    MPU6050_Init();
    
    initMotorPins(); //Initializes pins and hardware interupts for motors
    initRegParam(1, 0, 0, -0.0431, -0.0464);
    setupFirstValue();
}

int main(){
    setup();

    for(EVER){

        //Update IMU
        curTheta = update_angle(1);
        speed = calcSpeed(1);
        //Calc u 
        u =angleController(curTheta,0.0, 0.0);

        desPower = fabs(u*1024.0/12.0);
        if(u<0){
            dir = 0; //Maybe the other way? Test and see 
        }
        else{
            dir = 1;
        }

        accuateMotor(desPower,dir,desPower,dir);

        //Check for failure
        if(abs(curTheta)>15){
            accuateMotor(0,1,0,1);
            exit(1);
        }
        delay(10);
    }
}

