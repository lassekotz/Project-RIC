import numpy as np
import time
import tflite_runtime.interpreter as tflite
import sys
import cv2
from PIL import Image
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify




def initialize(resolution=128):
    model = str(sys.argv[1])
    if not model:
        raise RuntimeError("No model specified on sys.argv[1]")
    print(model)
    print("CAMERA INITIALIZING...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution)
    cap.set(cv2.CAP_PROP_FPS, 100)

    print("CAMERA INITIALIZED!")
    print("LOADING MODEL...")
    interpreter = edgetpu.make_interpreter(model_path_or_content="trained_models/" + model + "/" + model + "_full_int_quant_edgetpu.tflite")
    interpreter.allocate_tensors()
    #if common.input_details(interpreter, 'dtype') != np.uint8:
	    #raise ValueError("Only np uint8 is supported")
    size = common.input_size(interpreter)
    print(common.input_details(interpreter, 'quantization_parameters'))
    print("MODEL LOADED!")

    return cap, interpreter


def inference_step(interpreter, input_data, input_details, output_details):
    common.set_input(interpreter, input_data.astype(np.uint8))
    ti = time.time()
    interpreter.invoke()
    print(classify.get_classes(interpreter))

    return 1


def main(write_to_disk = False):
    if write_to_disk:
        f = open('timeseries.txt', 'w')
    resolution = 128
    cap, interpreter = initialize(resolution)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    i = 1
    tTot = 0
    while True:
        tt = time.time()
        ret, image = cap.read()
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #if not ret:
        #    raise RuntimeError("failed to read frame")
        image = image[:, :, [2, 1, 0]]
        #image = image/255.0
        input_data = np.array(image, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)
        #print(input_data)

        pred = inference_step(interpreter, input_data.astype(np.int8), input_details, output_details)
        #tTot = tTot + (time.time() - tt)
        #pred = a*(pred) + (1-a)*pred
        print(f'{pred:.2f}' + "\n")
        #p_prev = pred
        #
        #
        #
        #
        #

        if write_to_disk:
            f.write(str(pred) + "\n")
        if i%100 == 0:
            print(f'{1000*tTot/i}' + " ms/iter")
            tTot = 0
            i = 0
        i += 1
        # TODO: JIT
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main(write_to_disk=False)
