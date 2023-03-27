#include <stdio.h>
#include <cstdio>
#include <MPU6050.h>
#include <math.h>
#include <wiringPi.h>


MPU6050 mpu(0x68);
#define EVER ;;

int main() {
    float angle;

    sleep(1);
    
    for(EVER){
        mpu.getAngle(0,&angle);
        std::cout << "Curr angle= " << angle << "\n" << std::endl;
        delay(10);
    }
    
    return 0;
}

//g++ -o IMUtest IMUlib.cpp -l MPU6050 -lwiringPi -pthread