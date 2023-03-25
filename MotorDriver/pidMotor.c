#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>







unsigned long oldTcontroller;
double errsum = 0;
double oldErr,ITerm; //Error variables for angles 
double verrorSum, lastvRef; //Integral error for velocity controller 
float kp, ki, kd,Tf;
float kpv, kiv; //PI constans for velocity controller 
const double maxU = 6; //Maximum voltage we can send
const double minU = -6;  
//Motor regulator variables
double diffESum;
unsigned long oldTmR;
float kpm,kim,kdm;
float Ts; //Sample rate
float a;
float oldErrFilt = 0;


float angleController(float angle,float v, float vref){
   
   // Calculates output signals from inputs consisting of current leaning angle and desired angle

   // Keeping track of time
   u_int curTcontroller = millis();
   double dt = (double)(curTcontroller-oldTcontroller);

   //Velocity errors 
   double verror = vref-v;
   verrorSum += verror*dt;

   //Velocity PI controller 
   float angleRef = kpv*verror + kiv*verrorSum; //Acts as desired angle for next pid controller
   
   // Keep track of angle errors 
   double error = angleRef-angle;
   float eFilt = a*error+(1-a)*oldErrFilt;
   printf("a = %f \n",a);
   printf("error: %f   eFilt: %f \n",error,eFilt);
   double dErr = (eFilt-oldErrFilt)/dt;
   

   //Angle PID controller 
   ITerm += (ki * dt * error); //Prevent oversaturation of intergral error 
   if(ITerm> maxU){
      ITerm= maxU;}
   else if(ITerm< minU){
      ITerm= minU;}
   float u = kp*error + ITerm + kd*dErr;
   printf("Pterm: %f  iTERM: %f, kd %f\n",kp*error,ITerm,kd*dErr);

   // Store for next loop
   errsum += error*dt;
   oldTcontroller = curTcontroller;
   oldErrFilt = eFilt;
   
   return u;
}


float motorRegulator(float v1, float v2,float diffRef){
   //PID to control difference between the 2 motors

   // Keeping track of time
   u_int curTmR = millis();
   double dt = (double)(curTmR-oldTmR);

   //Velocity errors between motors
   double diffE = diffRef-(v1-v2);
   diffESum += diffE*dt;

   //Velocity PI controller 
   float u = kpm*diffE + kim*diffESum;

   //Store time for next loop
   oldTmR = curTmR;
   
   return u;
}

void initRegParam(float Kp, float Ki, float Kd, float Kpv, float Kiv){
   kp = Kp;
   ki = Ki;
   kd = Kd;
   kpv = Kpv;
   kiv = Kiv;
   a = 0.8;
}

