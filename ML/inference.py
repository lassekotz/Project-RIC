import numpy as np
import time
import tflite_runtime.interpreter as tflite

def run_inference():

    interpreter = tflite.Interpreter(model_path="trained_models/vgg16/vgg16.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    print(len(output_data))

t0 = time.time()
run_inference()
print("Elapsed time: " + str(time.time() - t0))
