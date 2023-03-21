import numpy as np
import time
import tflite_runtime.interpreter as tflite
import sys
import picamera
from picamera.array import PiRGBArray

def inference_step(interpreter, input_data):
    # Get input and output tensors.
    print(input_data)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on input data
    input_shape = input_details[0]['shape']
    #input_data = np.array(np.random.random_sample(input_shape))
    
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # The function 'get_tensor()' returns a copy of the tensor data.
    # Use 'tensor()' in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #output_data = output_data[0][0]
    print(output_data)

    return output_data

def initialize():
    model = str(sys.argv[1])
    print("CAMERA INITIALIZING...")
    camera = picamera.PiCamera()
    camera.resolution = (224, 224)
    camera.rotation = 180
    camera.framerate = 32
    camera.start_preview()
    time.sleep(2)
    camera.stop_preview()

    rawCapture = PiRGBArray(camera, size=camera.resolution)
    print("CAMERA INITIALIZED!")
    print("LOADING MODEL...")
    interpreter = tflite.Interpreter(model_path="trained_models/" + model + "/" + model + ".tflite")
    interpreter.allocate_tensors()
    print("MODEL LOADED!")

    return camera, interpreter, rawCapture

def main():
    camera, interpreter, rawCapture = initialize()
    t0 = time.time()

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                
        image = frame.array
        input_data = np.array(image, dtype='float32')
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.swapaxes(input_data, 1, 3)
                
        pred = inference_step(interpreter, input_data)
        #t1 = time.time()
        rawCapture.seek(0)
        rawCapture.truncate()
        #print(t1-t0)
        #t0 = t1
        
        
if __name__ == "__main__":
    main()
