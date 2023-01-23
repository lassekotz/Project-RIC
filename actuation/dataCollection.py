import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import importlib
import time as time
from BaseMotor import Motor
import sys
from picamera import PiCamera
from camera import Camera
from driveUpload import drive_upload
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def main(motor1, motor2, ang_lim, camera):
    IMU_datalist = []
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    folderURL = '1ldfh7h8yc_y4OBUwetNmSMDhMoZPCeIC?fbclid=IwAR07nNB8vSknC7Zf3IiOnHT6xpZux-ftqv00UjPydDV7ITjQBhTWc3TxXnE'

    file_list = drive.ListFile(
        {'q': "'{}' in parents and trashed=false".format(folderURL)}).GetList()
    j = len(file_list)

    imlist = []
    for i in range(round(ang_lim / 0.18)): #.18 is stepsize in degrees
        motor1.run(False)
        motor2.run(True)
        ext = '.jpg'
        imgname = 'picture%s%s' %(j, ext) #format image name string
        print(imgname)
        camera.capture(imgname)
        
        time.sleep(1)
        j += 1
        imlist.append(imgname)
        
        
    return imlist
        #TODO PRINT IMU-DATA
        #TODO CAPTURE IMAGE
        
        #IMU_datalist.append(imu_datapoint)


motor1 = Motor([24,25,8,7])
motor2 = Motor([])
camera = PiCamera()


imlist = main(motor1, motor2, 10, camera)
drive_upload(imlist)