// Include standard libraries  
#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>
#include <math.h>
// Include project libraries 
#include "motorControl.h"
#include "pidMotor.h"
#define EVER ;;


//float u = 0;
int inputPower = 0;
int dir;
int dir1;
float *speed;


PI_THREAD (motor_thread){
    for(EVER){

        if(inputPower<0){
            dir1 = 0; //Maybe the other way? Test and see 
        }
        else{
            dir1 = 1;
        }
        speed = calcSpeeds(0);
        float u = motorRegulator(speed[0], speed[1], 0);
        printf("speed 0  = %f and speed 1 = %f \n",speed[0],speed[1]);
        if(u<0){
            dir = 1; //Maybe the other way? Test and see 
        }
        else{
            dir = 0;
        }
        
        accuateMotor(abs(inputPower),dir1,abs(ceil(u)),dir);
        delay(10);
    }  
}

PI_THREAD (Input_thread){
    for(EVER){

    //Add encoder speed as second argument to function
    //Add desired speed as third argument to function
    printf("Input a value from 0-1024 \n");
    scanf("%d",&inputPower);
    if(inputPower == 1000){
        return(0);
    }
    delay(1000);
    }
}


int main(int argc, char *argv[] ){
    if( argc != 4 ){
        printf("Wrong number of arguments, you had %d \n",argc);
        exit(2);
    }
    float Kp = (float)atof(argv[1]);
    float Ki = (float)atof(argv[2]);
    float Kd = (float)atof(argv[3]);

    wiringPiSetupGpio(); //Setup and use defult pin numbering
    
    initMotorPins(); //Initializes pins and hardware interupts for motors
    initMotRegParam( Kp, Ki, Kd);
    piThreadCreate(Input_thread);
    piThreadCreate(motor_thread);
    
    for(EVER){
        delay(100);
    }
    return 0;
    }

