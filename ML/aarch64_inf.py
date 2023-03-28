import numpy as np
import time
import tflite_runtime.interpreter as tflite
import sys
import cv2


# from picamera.array import PiRGBArray


def initialize(resolution=128):
    model = str(sys.argv[1])
    if not model:
        raise RuntimeError("No model specified on sys.argv[1]")
    print(model)
    print("CAMERA INITIALIZING...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution)
    cap.set(cv2.CAP_PROP_FPS, 50)
    '''
    camera = picamera.PiCamera()
    camera.resolution = resolution
    camera.rotation = 180
    camera.framerate = 32
    camera.start_preview()
    time.sleep(3)
    camera.stop_preview()
	'''
    # rawCapture = PiRGBArray(camera, size=camera.resolution)
    print("CAMERA INITIALIZED!")
    print("LOADING MODEL...")
    interpreter = tflite.Interpreter(model_path="trained_models/" + model + "/" + model + "_quant.tflite")
    interpreter.allocate_tensors()
    print("MODEL LOADED!")

    return cap, interpreter


def inference_step(interpreter, input_data, input_details, output_details):
    # Test the model on input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # The function 'get_tensor()' returns a copy of the tensor data.
    # Use 'tensor()' in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


def main():
    resolution = 128
    cap, interpreter = initialize(resolution)
    t0 = time.time()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    while True:
        t0 = time.time()
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")
        image = image[:, :, [2, 1, 0]]
        input_data = np.array(image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.swapaxes(input_data, 1, 3)
        pred = inference_step(interpreter, input_data, input_details, output_details)
        print("iteration freq: " + str(1 / (time.time() - t0)) + "Hz")

        # TODO: JIT


if __name__ == "__main__":
    main()