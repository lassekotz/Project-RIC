from picamera import picamera
from time import sleep

camera = PiCamera()
camera.start_preview()
sleep(5)
camera.capture('picture.jpg')
camera.stop_preview()

