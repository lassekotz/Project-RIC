
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <wiringPi.h>
// Include project libraries 
#include "motorControl.h"

float curTheta;
float u = 0;
int desPower;
int dir;
double speed;
u_int32_t oldTid;
u_int32_t curTid;

int main()
{
    double uArray[1500];
    int rows = 0;
    char const* const fileName = "inputs.txt"; /* should check that argc > 1 */
    FILE* file = fopen(fileName, "r"); /* should check the result */
    char line[256];

    while (fgets(line, sizeof(line), file)) {
        /* note that fgets don't strip the terminating \n, checking its
           presence would allow to handle lines longer that sizeof(line) */
        uArray[rows] = atof(line);
        rows++;
        
    }

    fclose(file);

    wiringPiSetupGpio(); //Setup and use defult pin numbering
    initMotorPins(); //Initializes pins and hardware interupts for motors
    char *filename = "speeds.txt";

    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error opening the file %s", filename);
        return -1;
    }

    int i = 0;
    oldTid = millis();
    while(1){
        curTid = millis();
        double dtid = (curTid-oldTid)/1000.0;
        if(dtid >=0.1){
            if(i==rows){
                break;
            }

            u = uArray[i];
            desPower = fabs(u*1024.0/12.0);
            if(u<0){
                dir = 0; //Maybe the other way? Test and see 
            }
            else{
                dir = 1;
            }

            accuateMotor(desPower,dir,desPower,dir);
            speed = calcSpeed(0);

            fprintf(fp, "%f\n", speed);
            i++;
            oldTid = curTid;
        }
    }
    

    fclose(fp);
    

    return 0;
}

