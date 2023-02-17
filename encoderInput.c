#include <stdio.h>
#include <math.h>
#include <wiringPi.h>
#include <stdlib.h>

wiringPiSetupGpio(); //Setup and use defult pin numbering

// Declare pins
const int chA1 = 1; //Change these 
const int chB1 = 1;
const int chA2 = 1;  
const int chB2 = 1; 
int pos = 0;

const double alpha = 360.0/3.0/231.0;
pinMode(chA1,INPUT);
pinMode(chB1,INPUT);

void readEncoder(){
    int b = digitalRead(chB1);
    if(b >0){
        pos++;
    }
    else{
        pos--;
    }
}

int main(){

    wiringPiISR(chA1,INT_EDGE_RISING, &readEncoder);
    // wiringPiISR (0, INT_EDGE_FALLING, &myInterrupt0) ;
    // int waitForInterrupt (int pin, int timeOut) ;
    return 0;
}

