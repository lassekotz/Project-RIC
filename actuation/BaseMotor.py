import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib

class Motor():
<<<<<<< HEAD
    def __init__(self, pins):
        
=======
    def __init__(self):
        pass

    def initialize(self, pins) -> bool:
>>>>>>> be4848d18d33f5f144b9c1eb734f841d9d691b98
        self.pins = pins
        self.motor = RpiMotorLib.BYJMotor("Motor", "28BYJ")
        


    def run(self,nr_of_steps = 20, counterclockwise=False):
        '''
        Run the motor. 


        ARGS: (self, gpiopins, wait=.001, steps=512, ccwise=False,
                  verbose=False, steptype="half", initdelay=.001):
        '''
        # call the function pass the parameters
<<<<<<< HEAD
        self.motor.motor_run(self.pins, .01, nr_of_steps,
                             counterclockwise, False, "full", .01)
                        
        return nr_of_steps
=======
        self.motor.motor_run(self.pins, .01, 10,
                             counterclockwise, False, "full", .05)
>>>>>>> be4848d18d33f5f144b9c1eb734f841d9d691b98

    def shutdown(self) -> bool:
        '''
        Clean up the gpio pins.
        '''

        GPIO.cleanup()

        return True
        
