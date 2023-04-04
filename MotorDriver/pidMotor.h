#ifndef RIC_pid
#define RIC_pid

// Functions
float angleController(float angle,float v, float vref,int verbose);
void initRegParam(float Kp, float Ki, float Kd, float Kpv, float Kiv);
float motorRegulator(float v1, float v2,float diffRef);
void initMotRegParam(float Kpm, float Kim,float Kdm);

#endif

