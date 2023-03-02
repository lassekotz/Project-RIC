#ifndef RIC_motor
#define RIC_motor


// Functions used
void accuateMotor(int power1,int dir1,int power2,int dir2);
void readEncoder1();
void readEncoder2();
int initMotorPins();
void printWheelRotation();


#endif

