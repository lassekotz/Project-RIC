#ifndef RIC_pid
#define RIC_pid

// Functions
float angleController(float angle,float v, float vref);
void initRegParam(float Kp, float Ki, float Kd, float Kpv, float Kiv);
 

#endif

