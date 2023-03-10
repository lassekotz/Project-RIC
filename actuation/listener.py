import serial

ser = serial.Serial('/dev/ttyAMA0', 9600, timeout = 10)
while True:
	ser.read()
