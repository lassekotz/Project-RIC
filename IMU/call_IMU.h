#ifndef RIC_imu
#define RIC_imu

//Functions 
void MPU6050_Init();
short read_raw_data(int addr);
void ms_delay(int val);
double update_angle(int verbose);
void setupFirstValue();




#endif
