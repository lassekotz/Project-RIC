#ifndef RIC_pid
#define RIC_pid

// Functions
float angleController(float angle,float v, float vref);
void initRegParam(float Kp, float Ki, float Kd, float Kpv, float Kiv);
float motorRegulator(float v1, float v2,float diffRef);

#endif

