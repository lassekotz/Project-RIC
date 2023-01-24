import serial
import time
'''
def send_steps(steps):
    ser = serial.Serial('/dev/ttyACM0', 9600)
    readedText = ser.readline()
    print(readedText)
    ser.close()
'''

msg = 5
ser = serial.Serial('COM3', 9600)
ser.write(bytes(msg))
time.sleep(.5)
a = ser.readline()
print(a.decode('UTF-8'))

ser.close()
