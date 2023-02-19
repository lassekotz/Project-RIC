#include <stdio.h>
#include <math.h>
#include <wiringPi.h>
#include <stdlib.h>



// Declare pins
const int chA1 = 21; //Change these 
const int chB1 = 20;
//const int chA2 = 1;  
//const int chB2 = 1; 
double pos = 0;

const double alpha = 360.0/3.0/231.0;


void readEncoder(){ 
    int b = digitalRead(chB1);
    if(b >0){
        pos= pos+alpha;
    }
    else{
        pos = pos-alpha;
    }
    printf("Current angle is %f",pos);
}

int main(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering
    pinMode(chA1,INPUT);
    pinMode(chB1,INPUT);
    printf("The pos is: %f",pos);
    wiringPiISR(chA1,INT_EDGE_RISING, &readEncoder); //Hardware interupt on encoder pin
    while(1){
        delay(500);
        printf("\n");
    }
    // wiringPiISR (0, INT_EDGE_FALLING, &myInterrupt0) ;
    // int waitForInterrupt (int pin, int timeOut) ;
    return 0;
}

//gcc encoderInput.c -o EN -lwiringPi -lm


