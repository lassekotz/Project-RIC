#ifndef RIC_motor
#define RIC_motor


// Functions used
void accuateMotor(int power1,int dir1,int power2,int dir2);
void readEncoder1();
void readEncoder2();
int initMotorPins();
void printWheelRotation();

// Declaring pins for motors
const int en1; //Change these 
const int in1; 
const int in2;
const int in3;
const int in4;
const int en2;
// Declaring pins for encoders
const int chA1; //Change these 
const int chB1;
const int chA2;  
const int chB2; 
double pos1; //Keeps track of current angle 
double pos2;


#endif

const double alpha; // Relative angle per encoder step
u_int currT; // Current time in milliseconds 
u_int oldT;