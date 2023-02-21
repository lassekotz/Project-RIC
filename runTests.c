#include "motorControl.h"
#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>


int main(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering

    //Setting pinmodes for motor pins
    pinMode(en1,PWM_OUTPUT);
    pinMode(en2,PWM_OUTPUT);
    pinMode(in1,OUTPUT);
    pinMode(in2,OUTPUT);
    pinMode(in3,OUTPUT);
    pinMode(in4,OUTPUT);
    //Pins for encoder 
    pinMode(chA1,INPUT);
    pinMode(chB1,INPUT);


    
    wiringPiISR(chA1,INT_EDGE_RISING, &readEncoder1); // Hardware interupt for encoder 1
    wiringPiISR(chA2,INT_EDGE_RISING, &readEncoder2);

}