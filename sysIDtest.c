#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <wiringPi.h>
// Include project libraries 
#include "motorControl.h"


float uArray[] = {0, //White noise generated to estimate motor parameters
0,
0,
0,
-6.53299722148481, 
-10.1908472987738,
-7.74215770931184,
3.27376046242883,
12.8120972145438,
-3.46115247201075,
4.95113432209721,
9.67862493361173,
7.29553309402658,
9.68346781685412,
-3.59963281901619,
6.24048005437827,
-1.46644104825758,
-8.91040049790521,
-2.13707176209416,
2.12714135466353,
11.0094772458562,
5.38615480772669,
1.51381380186363,
5.90416701783824,
-3.08736554669575,
-7.86392181094658,
-4.12389398409737,
-7.11278771550428,
5.89433167148658,
-0.812079410092151,
-7.14023980628917,
-1.24385774926460,
-5.07369111815235,
-1.88577076267173,
8.54568186809899,
8.94886112484826,
-4.46429035603874,
-2.24626188441757,
2.54464134488422,
0.684694252748329,
-3.06658774562750,
6.13402332361410,
-8.51510305953236,
15.2800759928742,
-2.14773301374655,
2.40791056031936,
-4.23716897805445,
-6.12398036928491,
6.80376497946714,
0.231271992160699,
1.38547635873944,
0.754401122046684,
4.17771367924944,
10.4368419088501,
3.99226431359671,
4.12519082923018,
7.43381229174520,
5.39197840568210,
-6.90058846319389,
-1.58508277764356,
-9.75108271398994,
3.48985559391412,
11.4149563743095,
9.48056565862687,
5.52617432988675,
-5.62741476129864,
-4.42311726823965,
1.07868471605077,
-0.790982844440056,
5.20756136079183,
7.88635130651977,
-1.50671730997694,
-0.430557072822715,
6.57125226597131,
2.93568220669182,
-10.0809221830864,
-4.86915275807727,
-4.51334934487959,
-7.95551455748032,
-1.08888512990307,
0.788492921736479,
8.36297969304258,
4.23728053666681,
6.97708086389852,
-4.52212634224387,
1.93154821377612,
9.81422521257304,
-3.88676713736898,
4.03957981239357,
7.64133299483281,
-6.23887092829508,
2.92770015866363,
-5.44052162456258,
-2.84821844992742,
-3.99652036776120,
-3.01211769094247,
-1.85331611894491,
-5.02140777983887,
-3.76994336260519,
3.00542074620742,
10.0215216735745,
-9.25038061223997,
-1.63205473284795,
2.04988042570466,
0,
0,
0,
0
};

float curTheta;
float u = 0;
int desPower;
int dir;
double speed;


int main(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering
    initMotorPins(); //Initializes pins and hardware interupts for motors

    char *filename = "speeds.txt";

    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error opening the file %s", filename);
        return -1;
    }

    for(int i= 0;i<sizeof(uArray)/sizeof(uArray[0]);i++){
        
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

        delay(100);
    }

    fclose(fp);

    return 0; 

    

}

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
u_int32_t oldT;
u_int32_t curT;

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
    oldT = millis();
    while(1){
        curT = millis();
        double dt = (curT-oldT)/1000.0;
        if(dt >=0.1){
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
            oldT = curT;
        }
    }
    

    fclose(fp);
    

    return 0;
}

