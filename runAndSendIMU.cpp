// Include standard libraries  
#include <stdio.h>
#include <wiringPi.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h> 
#include <math.h>
#include <MPU6050.h>
#include "Kalman.h"
#define EVER ;;
#define RAD_TO_DEG 57.2957795;

MPU6050 imu(0x68,0);
Kalman kalman;
const float TIMU = 0.01;

int main(){

    
    // Create or open the shared memory segment
    key_t key = 1234; //Match with python script
    int shmid = shmget(key, sizeof(float), 0666 | IPC_CREAT | IPC_EXCL);
    if (shmid == -1) {
        // If the segment already exists, just open it
        shmid = shmget(key, sizeof(float), 0666);
        if (shmid == -1) {
            perror("shmget");
            exit(1);
        }
    }

    // Attach the shared memory segment to the process' address space
    float *curTheta = (float*)shmat(shmid, NULL, 0);
    if (curTheta == (float*)-1) {
        perror("shmat");
        exit(1);
    }

    // Init Kalman filter
    imu.getAngle(0,&curTheta);
    kalman.setAngle(curTheta); 

    for(EVER){

        curTime = millis();
        float dtIMU = (curTime-lastIMUtime)/1000.0f;

        if(dtIMU>=TIMU){

            imu.getGyro(&gr, &gp, &gy);
            imu.getAccel(&accX, &accY, &accZ);
            double roll  = atan(accY / sqrt(accX * accX + accZ * accZ)) * RAD_TO_DEG;
            *curTheta = -kalman.getAngle(roll, gr, dtIMU);

        }

    }
}