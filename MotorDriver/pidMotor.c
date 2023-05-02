#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>







unsigned long oldTcontroller;
double errsum = 0;
double oldErr,ITerm; //Error variables for angles 
double verrorSum, lastvRef; //Integral error for velocity controller 
float kp, ki, kd,Tf;
float kpv, kiv; //PI constans for velocity controller 
const double maxU = 700; //Maximum PWM we can send through integral accumulation
const double minU = -700;  
//Motor regulator variables
double diffESum;
double OldDiffE = 0;
unsigned long oldTmR;
float kpm,kim,kdm; // 10 10 10 seems to work good 
float Ts; //Sample rate
float a;
float oldErrFilt = 0;


float angleController(float angle,float v, float vref,int verbose){
   
   // Calculates output signals from inputs consisting of current leaning angle and desired angle

   // Keeping track of time
   u_int curTcontroller = millis();
   double dt = (double)(curTcontroller-oldTcontroller)/1000.0;
   //Velocity errors 
   double verror = vref-v;
   verrorSum += verror*dt;

   //Velocity PI controller 
   float angleRef = kpv*verror + kiv*verrorSum; //Acts as desired angle for next pid controller
   
   angleRef = angleRef-0.3;
   
   // Keep track of angle errors 
   double error = angleRef-angle;
   float eFilt = a*angle+(1-a)*oldErrFilt; //Low Pass filter on meassurment
   //printf("a = %f \n",a);
   //printf("error: %f   eFilt: %f \n",error,eFilt);
   double dErr = (eFilt-oldErrFilt)/dt;
   //printf("Filtered error = %f \n",dErr);
   
   //Angle PID controller 
   ITerm += (ki * dt * error); //Prevent oversaturation of intergral error 
   if(ITerm> maxU){
      ITerm= maxU;}
   else if(ITerm< minU){
      ITerm= minU;}

   
   float u = kp*error + ITerm + kd*dErr;
   if(error>15){
      u = kp*1.9*error + ITerm + kd*dErr;  
   }
   if(verbose){
      printf("Pterm: %f  iTERM: %f, kd %f\n",kp*error,ITerm,kd*dErr);
      printf("angleRef: %f \n",angleRef);
   }
   

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
   double dt = (double)(curTmR-oldTmR)/1000.0;

   //Velocity errors between motors
   double diffE = diffRef-(v1-v2);
   diffESum += diffE*dt;
   double diffEder = (diffE-OldDiffE)/dt; 

   //Velocity PI controller 
   float u = kpm*diffE + kim*diffESum+kdm*diffEder;

   //Store time for next loop
   oldTmR = curTmR;
   OldDiffE = diffE;
   return u;
}

void initMotRegParam(float Kpm, float Kim,float Kdm){
   kpm = Kpm;
   kim = Kim;
   kdm = Kdm;
}

void initRegParam(float Kp, float Ki, float Kd, float Kpv, float Kiv){
   kp = Kp;
   ki = Ki;
   kd = Kd;
   kpv = Kpv;
   kiv = Kiv;
   a = 0.4;
}

