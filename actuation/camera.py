#from device.base_camera import BaseCamera
#import util

from picamera import PiCamera
from picamera.array import PiRGBArray


class Camera(): #instantiate camera
    '''
    Implementation of the base camera interface for the raspberry pi
    V2 camera module.
    '''

    def __init__(self) -> None:
        pass

    def initialize(self) -> bool:
        '''
        Initializes the camera.
        '''
        self.camera = PiCamera()
        return True

    def shutdown(self) -> bool:
        '''
        Shuts down the video feed.
        '''
        self.camera = None

        return True

    def capture_image(self, encode_jpg=True):
        '''
        Performs capture from the camera.
        '''
        raw_capture = PiRGBArray(self.camera)

        #self.camera.capture(raw_capture, format="bgr") #SEND TO FILE
        image = raw_capture.array

        self.camera.capture('picture.jpg')

        #return image
