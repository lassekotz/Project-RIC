import sys
from BaseMotor import Motor

steps = int(sys.argv[1])
rotation = sys.argv[2]
motor = Motor()

motor.run(steps, True)