import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib

class Motor():
    def __init__(self):
        pass

    def initialize(self, pins) -> bool:
        self.pins = pins
        self.motor = RpiMotorLib.BYJMotor("Motor", "28BYJ")

    def run(self, counterclockwise=False):
        '''
        Run the motor. 


        ARGS: (self, gpiopins, wait=.001, steps=512, ccwise=False,
                  verbose=False, steptype="half", initdelay=.001):
        '''
        # call the function pass the parameters
        self.motor.motor_run(self.pins, .01, 10,
                             counterclockwise, False, "full", .05)

    def shutdown(self) -> bool:
        '''
        Clean up the gpio pins.
        '''

        GPIO.cleanup()

        return True
        
