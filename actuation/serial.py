import serial

def send_steps(steps):
    ser = serial.Serial('/dev/ttyACM0', 9600)
    readedText = ser.readline()
    print(readedText)
    ser.close()