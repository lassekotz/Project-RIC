import numpy as np
import time
import tflite_runtime.interpreter as tflite
import sys
import picamera
from picamera.array import PiRGBArray


def initialize(resolution=(128, 128)):
    model = str(sys.argv[1])
    print(model)
    print("CAMERA INITIALIZING...")
    camera = picamera.PiCamera()
    camera.resolution = resolution
    camera.rotation = 180
    camera.framerate = 32
    camera.start_preview()
    time.sleep(3)
    camera.stop_preview()

    rawCapture = PiRGBArray(camera, size=camera.resolution)
    print("CAMERA INITIALIZED!")
    print("LOADING MODEL...")
    interpreter = tflite.Interpreter(model_path="trained_models/" + model + "/" + model + "_quant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print("MODEL LOADED!")

    return camera, interpreter, rawCapture, input_details, output_details
    
def inference_step(interpreter, input_data, input_details, output_details):
    # Get input and output tensors.


    # Test the model on input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # The function 'get_tensor()' returns a copy of the tensor data.
    # Use 'tensor()' in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data



def main():
    resolution = (128, 128)
    camera, interpreter, rawCapture, input_details, output_details = initialize(resolution)
    t0 = time.time()
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        input_data = np.array(image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.swapaxes(input_data, 1, 3)
        pred = inference_step(interpreter, input_data, input_details, output_details)
        rawCapture.seek(0)
        rawCapture.truncate()
        print("Sample rate: " + str(1/(time.time() - t0)) + " Hz")
        t0 = time.time()

        # TODO: JIT

        
        
if __name__ == "__main__":
    main()
