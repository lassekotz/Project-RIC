import numpy as np
import time
import tflite_runtime.interpreter as tflite
import sys
import picamera
from picamera.array PiRGBArray

def inference_step(interpreter, input_data):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on input data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape))

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # The function 'get_tensor()' returns a copy of the tensor data.
    # Use 'tensor()' in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

def initialize():
    model = str(sys.argv[1])
    print("CAMERA INITIALIZING...")
    camera = picamera.PiCamera()
    camera.resolution = (128, 128)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    print("CAMERA INITIALIZED!")
    print("LOADING MODEL...")
    interpreter = tflite.Interpreter(model_path="trained_models/" + model + "/" + model + ".tflite")
    interpreter.allocate_tensors()
    print("MODEL LOADED!")

    return camera, interpreter, rawCapture

def main():
    camera, interpreter, rawCapture = initialize()

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        input_data = np.array(image)
        pred = inference_step(interpreter, input_data)


