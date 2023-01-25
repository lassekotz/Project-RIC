#from device.base_camera import BaseCamera
#import util

from picamera import PiCamera
from picamera.array import PiRGBArray


class Camera(): #instantiate camera
    '''
    Implementation of the base camera interface for the raspberry pi
    V2.1 camera module.
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

    def capture_image(self, img_number, resolution = (1920, 1080)):
        '''
        Performs capture from the camera.
        '''

        ext = '.jpg'
        imgname = 'Images/picture%s%s' % (img_number, ext)  # format image name string
        print(imgname)
        self.camera.resolution = resolution
        self.camera.capture(imgname)

        #return image