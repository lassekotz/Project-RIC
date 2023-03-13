// Include standard libraries  
#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>
#include <math.h>
// Include project libraries 
#include "motorControl.h"


int main(){
	wiringPiSetupGpio();
	initMotorPins();
	accuateMotor(0,0,0,0);
	return 0;
}



//gcc turnOff.c -o panicStop -lwiringPi -lm
