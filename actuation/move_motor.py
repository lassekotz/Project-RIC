import sys
from BaseMotor import Motor

steps = str(sys.argv[1])
rotation = sys.argv[2]
motor = Motor()

motor.run(steps, True)