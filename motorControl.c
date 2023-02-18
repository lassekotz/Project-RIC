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
int pos = 0;

wiringPiSetupGpio(); //Setup and use defult pin numbering

pinMode(en1,PWM_OUTPUT);
pinMode(en2,PWM_OUTPUT);
pinMode(in1,OUTPUT);
pinMode(in2,OUTPUT);
pinMode(in3,OUTPUT);
pinMode(in4,OUTPUT);


const double alpha = 360.0/3.0/231.0; // Relative angle per encoder step
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

int main(){
    u_int currT = millis()
    wiringPiISR(chA1,INT_EDGE_RISING, &readEncoder);

}
n 