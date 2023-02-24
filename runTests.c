
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


    printf("Initializing IMU and regulator");
    initRegParam(28.545755616, 241.5669, 2.4835, -0.0431, -0.0464);
    setupFirstValue();
    float curTheta;
    float u;
    for(int i; i<1000; i++){
        curTheta = update_angle(1);
        u =angleController(curTheta,0.0, 0.0);
        printf("Desired motor voltage from controller %f",u);
        delay(10);
    }
 
}

/*Controllers.F_theta.Kp = 28.545755616786778;
Controllers.F_theta.Ki = 241.5669;
Controllers.F_theta.Kd = 2.4835;
Controllers.F_theta.Tf = 0.0400;

% Yttre regulatorn 1: F_v
Controllers.F_v.Kp = -0.0431;
Controllers.F_v.Ki = -0.0464;*/ 
