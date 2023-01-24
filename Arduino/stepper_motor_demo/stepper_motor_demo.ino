//Includes the Arduino Stepper Library
#include <Stepper.h>

// Defines the number of steps per rotation
const int stepsPerRevolution = 2038;

// Creates an instance of stepper class
// Pins entered in sequence IN1-IN3-IN2-IN4 for proper step sequence
Stepper motor1(stepsPerRevolution, 8, 10, 9, 11);
Stepper motor2(stepsPerRevolution, 4,6,5,7); 

int process[] = {300,-300,500,-500,100,-100};
int j = 0;


// Choose speed



void setup() {
	// Nothing to do (Stepper Library sets pins as outputs)
  motor1.setSpeed(5);
  motor2.setSpeed(5);
}

void loop() {
	// Rotate CW slowly at 5 RPM
	move_steps(process[j]);
  j++;
  if(j==6){
    j = 0;
  }
  delay(1000);
}

void move_steps(int num_steps){
  int i = 0;
  if(num_steps>0){
    while(i<abs(num_steps)){
      motor1.step(1); //Move motor1 1 step
      motor2.step(1); 
      i++;
    }
  }
  else{
    while(i<abs(num_steps)){
      motor1.step(-1); //Move motor1 1 step
      motor2.step(-1); 
      i++;
    }
  }
}





