#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>
#include <math.h>
//#include "motorControl.h"

// Declaring pins for motors
const int en1 = 12; //Change these 
const int in1 = 14; 
const int in2 = 15;
const int in3 = 8;
const int in4 = 7;
const int en2 = 13;
// Declaring pins for encoders
const int chA1 = 26; //Change these 
const int chB1 = 19;
const int chA2 = 21;  
const int chB2 = 20; 
double pos1 = 0; //Keeps track of current angle 
double pos2 = 0;
double oldPos1=0;
double oldPos2 =0;
double curSpeed = 0;




const double alpha = 360.0/3.0/231.0; // Relative angle per encoder step
u_int currT = 0; // Current time in milliseconds 
u_int oldT = 0;
// Speed tracking variables 
u_int curTv = 0; 
u_int oldTv = 0;
double oldV = 0;
double dtv = 0;
double v = 0; 

void readEncoder1(){ 
    int b = digitalRead(chB1);
    if(b >0){
        pos1 = pos1+alpha;
    }
    else{
        pos1 = pos1-alpha;
    }

}

void readEncoder2(){ 
    int b = digitalRead(chB2);
    if(b >0){
        pos2 = pos2+alpha;
    }
    else{
        pos2 = pos2-alpha;
    }
}

void printWheelRotation(){
    //Only used for testing 
    printf("Wheel 1 has rotated %f degrees \n",pos1);
    printf("Wheel 2 has rotated %f degrees \n",pos2);
}

double calcSpeed(int verbose){
    curTv = millis();
    dtv = (curTv-oldTv)/1000.0;
    curSpeed = ((pos1-oldPos1)+(pos2-oldPos2))/(2.0*dtv);
    
    //Probably needs to be low pass filtered ?
    if(verbose){
        printf("Current speed %f deg/sec \n",curSpeed);
    }
    oldPos1 = pos1;
    oldPos2 = pos2;
    return curSpeed;
       
}

/*
float* convolve(float h[], float x[], int lenH, int lenX, int* lenY)
{
  int nconv = lenH+lenX-1;
  (*lenY) = nconv;
  int i,j,h_start,x_start,x_end;

  float *y = (float*) calloc(nconv, sizeof(float));

  for (i=0; i<nconv; i++)
  {
    x_start = max(0,i-lenH+1);
    x_end   = min(i+1,lenX);
    h_start = min(i,lenH-1);
    for(j=x_start; j<x_end; j++)
    {
      y[i] += h[h_start--]*x[j];
    }
  }
  return y;
} */


void accuateMotor(int power1,int dir1,int power2,int dir2){ 
    //Dir = Zero for backwards, One for forwards
    // Power is a number between 0-1024
    
    if(power1 < 300 && fabs(curSpeed)<0.1){ //
        accuateMotor(300,dir1,300,dir2);
        delay(10);
    }
    // Direction motor 1
    if(dir1 == 0){
        digitalWrite(in1,1);
        digitalWrite(in2,0);
    } else {
        digitalWrite(in1,0);
        digitalWrite(in2,1); 
    }

    // Direction motor 2
    if(dir2 == 0){
        digitalWrite(in3,1);
        digitalWrite(in4,0);
    } else {
        digitalWrite(in3,0);
        digitalWrite(in4,1);
    }

    pwmWrite(en1,power1);
    pwmWrite(en2,power2);

}

int initMotorPins(){
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


    // Hardware interupt for encoders
    wiringPiISR(chA1,INT_EDGE_RISING, &readEncoder1); 
    wiringPiISR(chA2,INT_EDGE_RISING, &readEncoder2);
    return 0;
}

/*
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

} */ 

//gcc motorControl.c -o mC -lwiringPi -lm
