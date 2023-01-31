//Includes the Arduino Stepper Library
#include <Stepper.h>

// Defines the number of steps per rotation
const int stepsPerRevolution = 2038;

// Creates an instance of stepper class
// Pins entered in sequence IN1-IN3-IN2-IN4 for proper step sequence
Stepper motor1(stepsPerRevolution, 8, 10, 9, 11);
Stepper motor2(stepsPerRevolution, 4,6,5,7); 

//int process[] = {300,-300,500,-500,100,-100};
//int j = 0;





void setup() {
	// Nothing to do (Stepper Library sets pins as outputs)
  motor1.setSpeed(5);
  motor2.setSpeed(5);
  Serial.begin(9600);
}

void loop() {

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

void serialEvent()
{
   while(Serial.available()) 
   {
      String desAngle = Serial.readString();
      Serial.println(desAngle.toInt());
      if (abs(desAngle.toInt()) < 1000)
      {
        move_steps(desAngle.toInt());
     }
   }
}








