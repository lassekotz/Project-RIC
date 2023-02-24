#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>







unsigned long oldT;
double errsum, oldErr,ITerm; //Error variables for angles 
double verrorSum, lastvRef; //Integral error for velocity controller 
float kp, ki, kd;
float kpv, kiv; //PI constans for velocity controller 
const double maxU = 12; //Maximum voltage we can send
const double minU = -12;  

float angleController(float angle,float v, float vref){
   // Calculates output signals from inputs consisting of current leaning angle and desired angle

   // Keeping track of time
   u_int curT = millis();
   double dt = (double)(curT-oldT);

   //Velocity errors 
   double verror = vref-v;
   verrorSum += verror*dt;

   //Velocity PI controller 
   float angleRef = kpv*verror + kiv*verrorSum; //Acts as desired angle for next pid controller

   // Keep track of angle errors 
   double error = angleRef-angle;
   errsum += error*dt;
   double dErr = (error-oldErr)/dt;


   //Angle PID controller 
   ITerm += (ki * errsum); //Prevent oversaturation of intergral error 
   if(ITerm> maxU){
      ITerm= maxU;}
   else if(ITerm< minU){
      ITerm= minU;}
   float u = kp*error + ITerm + kd*dErr;

   // Store for next loop
   oldT = curT;
   oldErr = error;
   
   return u;
}


initRegParam(float Kp, float Ki, float Kd, float Kpv, float Kiv;){
   kp = Kp;
   ki = Ki;
   kd = Kd;
   kpv = Kpv;
   kiv = Kiv;
}