import picamera
from picamera.array import PiRGBArray

def run_inference_on_live():
    with picamera.PiCamera() as camera:
        camera.resolution = (128, 128)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=camera.resolution)

    # TODO: load tflite model

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        #TODO: inference on image

        # pred = model(image)

        print(pred)



