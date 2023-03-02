#ifndef RIC_imu
#define RIC_imu

//Functions 
void MPU6050_Init();
short read_raw_data(int addr);
void ms_delay(int val);
double update_angle(int verbose);
void setupFirstValue();


//Declarations 
int fd;
float Acc_x,Acc_y,Acc_z;
float Gyro_x,Gyro_y,Gyro_z;
float Ax, Ay, Az;
float Gx, Gy, Gz;
double theta,thetaA,thetaG,thetaOld;
unsigned long oldTimu;

#endif