
// Functions
float angleController(float angle,float v, float vref);


// Declarations 
unsigned long oldT;
double errsum, oldErr,ITerm; //Error variables for angles 
double verrorSum, lastvRef; //Integral error for velocity controller 
float kp, ki, kd;
float kpv, kiv; //PI constans for velocity controller 
double maxU; //Maximum voltage we can send
double minU;  