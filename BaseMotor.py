class Motor(): # REPLACE WITH BaseMotor and seperate
    def __init__(PINOUT):
        pass #self.PINOUT = PINOUT


    def initialize():
        self.pins = [24, 25, 8, 7]

        # Declare an named instance of class pass a name and motor type
        self.motor = RpiMotorLib.BYJMotor("Motor", "28BYJ")

        return True

    def run(self, counterclockwise=False):
        '''
        Run the motor. 


        ARGS: (self, gpiopins, wait=.001, steps=512, ccwise=False,
                  verbose=False, steptype="half", initdelay=.001):
        '''
        # call the function pass the parameters
        self.motor.motor_run(self.pins, .01, 5,
                             counterclockwise, False, "full", .05)

    def shutdown(self) -> bool:
        '''
        Clean up the gpio pins.
        '''

        GPIO.cleanup()

        return True
        