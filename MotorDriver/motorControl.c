#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>

// Declaring pins for motors
const int en1 = 12; //Change these 
const int in1 = 23; 
const int in2 = 24;
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




const double alpha = 360.0/3.0/231.0; // Relative angle per encoder step
u_int currT = 0; // Current time in milliseconds 
u_int oldT = 0;

void readEncoder1(){ 
    int b = digitalRead(chB1);
    if(b >0){
        pos1 = pos1+alpha;
    }
    else{
        pos1 = pos1-alpha;
    }
    printf("Current angle is %f",pos);
}

void readEncoder2(){ 
    int b = digitalRead(chB2);
    if(b >0){
        pos2 = pos2+alpha;
    }
    else{
        pos2 = pos2-alpha;
    }
    printf("Current angle 2 is %f",pos);
}


int accuateMotor(int power1,int dir1,int power2,int dir2){ 
    //Dir = Zero for backwards, One for forwards
    // Power is a number between 0-1024

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
    pmwWrite(en2,power2);


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

} /* 

//gcc motorControl.c -o mC -lwiringPi -lm