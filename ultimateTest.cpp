// Include standard libraries  
#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>
#include <math.h>
#include <MPU6050.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
// Include project libraries 
extern "C" {
#include "motorControl.h"
}
extern "C" {
#include "pidMotor.h"
}
#include "Kalman.h"
#define EVER ;;
#define RAD_TO_DEG 57.2957795;

float curTheta;
float predictedTheta;
float kalmanTheta;
float kalmanAcc;
float accTheta;
float gyroTheta = 0;
float thetaG = 0;
float u = 0;
int desPower;
int dir,dir2;
float* speed;
MPU6050 imu(0x68,0);
Kalman kalman;
Kalman kalmanML;
double kalTheta;
float gr, gp, gy;
float accX, accY, accZ;

// Sampling times
const float TIMU = 0.1;
const float Tpid = 0.005;

unsigned long curTime;
unsigned long lastIMUtime, lastmotorTime, lastpidTime;

// Motor syncing
float uM2;

void setup(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering

    imu.getAngle(0,&curTheta); //Calculate first value and input to filter 
    kalman.setAngle(curTheta); 
    

    initMotorPins(); //Initializes pins and hardware interupts for motors
    //setupFirstValue();
}

int main( int argc, char *argv[] ){

    if( argc != 6 ){
        printf("Wrong number of arguments, you had %d \n",argc);
        exit(2);
    }
    float Kp = (float)atof(argv[1]);
    float Ki = (float)atof(argv[2]);
    float Kd = (float)atof(argv[3]);
    float Kpv = (float)atof(argv[4]);
    float Kiv = (float)atof(argv[5]);
    initRegParam( Kp , Ki, Kd, Kpv, Kiv);
    initMotRegParam( 3000.0, 30.0, 300.0);

    setup();

    // Open the named shared memory region
    int shared_memory_fd = shm_open("currAngle", O_RDONLY, 0666);
    if (shared_memory_fd == -1) {
        std::cerr << "Failed to open shared memory object\n";
        return 1;
    }
    

    // Map the shared memory region into this process's address space
    void* shared_memory_ptr = mmap(NULL, sizeof(float), PROT_READ, MAP_SHARED, shared_memory_fd, 0);
    if (shared_memory_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory object\n";
        close(shared_memory_fd);
        return 1;
    }

    lastIMUtime = millis();
    std::cout << "Starting up " << std::endl;
    delay(100);
    char const* const fileName = "ultimateTest.txt"; 
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL)
        {
            printf("Error opening the file %s", fileName);
            return -1;
        }
    int p = 0;
    for(EVER){

        curTime = millis();
        float dtIMU = (curTime-lastIMUtime)/1000.0f;
        if(dtIMU>=TIMU){
            //Update IMU

            
            //imu.getAngle(0,&curTheta); //Uncomment to use complementary filter
            //printf("Angle= %f \n",curTheta);
            
            /*
            //Kalman filter
            
            imu.getGyro(&gr, &gp, &gy);
            imu.getAccel(&accX, &accY, &accZ);
            double roll  = atan(accY / sqrt(accX * accX + accZ * accZ)) * RAD_TO_DEG;
            curTheta = -kalman.getAngle(roll, gr, dtIMU);
            */

            //Read from shared memory
            predictedTheta = *((float*)shared_memory_ptr);
            
            std::cout << "Prediction = "<< predictedTheta << std::endl;

            //Keep track of last time used
            lastIMUtime = curTime;
            if(dtIMU> TIMU*1.1){
               printf("Too slow time = %f \n",dtIMU);
            }
            
            
            fprintf(fp, "%f,%f,%f,%f,%f,%f\n", curTheta, predictedTheta, accTheta, gyroTheta, kalmanAcc, kalmanTheta);
            
            if (p > 100)
            {
                break;
            }
            p++ ;
            
        }

        float dtPID = (curTime-lastpidTime)/1000.0f;
        if(dtPID>=Tpid){
            //Update sensor values
            imu.getGyro(&gr, &gp, &gy);
            imu.getAccel(&accX, &accY, &accZ);

            //Calculate angle from accelerometer and gyros
            accTheta  = atan(accY / sqrt(accX * accX + accZ * accZ)) * RAD_TO_DEG;
            gyroTheta = gyroTheta + gr*dtPID;


            thetaG = curTheta + gr*dtPID; // deg/s * s = deg

		    // Complementary filter
		    curTheta = 0.98 * thetaG + 0.02 * accTheta;
            
            //Kalman filtering 
            kalmanAcc = -kalman.getAngle(accTheta, gr, dtPID); //Kalman with accelerometer and gyro 
            kalmanTheta = -kalmanML.getAngle(*((float*)shared_memory_ptr),gr,dtPID);
        }
        
        




        //Check for failure
        if(abs(curTheta)>25){
            accuateMotor(0,1,0,1);
            free(speed);
            accuateMotor(0,1,0,1);
            delay(10);
            accuateMotor(0,1,0,1);
            munmap(shared_memory_ptr, sizeof(float));
            close(shared_memory_fd);
            accuateMotor(0,1,0,1);
            fclose(fp);
            exit(1);
        }
    }
    
}


