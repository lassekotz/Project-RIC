
// Include standard libraries  
#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>
// Include project libraries 
#include "motorControl.h"
#include "pidMotor.h"
#include "call_IMU.h"


int main(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering
    MPU6050_Init();

    initMotorPins(); //Initializes pins and hardware interupts for motors

    printf("Start motor test 1 \n");
    accuateMotor(500,1,500,1);
    printf("Motors should now run at half speed forwards for 3 seconds");
    delay(3000);
    printf("Stop motors");
    accuateMotor(0,1,0,1);
    delay(100);
    accuateMotor(500,0,500,0);
    printf("Motors should now run at half speed backwards for 3 seconds");
    delay(3000);
    accuateMotor(0,1,0,1);
    printWheelRotation();
    delay(2000);
    printf("Initializing IMU");
    setupFirstValue();
    for(int i, i<1000,i++){
        update_angle(1);
        delay(10);
    }

    
}