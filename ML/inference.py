import numpy as np
import time
import tflite_runtime.interpreter as tflite
import sys
import torchvision.models as models
from model import CNN, LinearModel
from torch import nn

architecture = str(sys.argv[1])
extension = str(sys.argv[2]) #either tflite or pt

def run_inference_tflite(architecture):
    print("inference running on: " + architecture)
    interpreter = tflite.Interpreter(model_path="trained_models/" + architecture + "/" + architecture + ".tflite")
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

def run_inference_regular(architecture):
    #TODO: implement inference run from .pt file
    if architecture == 'linear':
        model = LinearModel()
        #TODO: instantiate linear and load parameters onto it
    elif architecture == 'CNN':
        model = CNN()
        #TODO: instantiate CNN and load params onto it
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)
        #TODO: isntantiate vgg16 and load parasms onto it
    else:
        raise Exception("Model " + architecture + " not available in ./trained_models/")
    model.load_state_dict('./trained_models/' + architecture + "/" + architecture + ".pt")

    pass

t0 = time.time()
if extension == '.pt':
    run_inference_regular(architecture)
    print("Elapsed time: " + str(time.time() - t0))
elif extension == '.tflite':
    run_inference_tflite()
    print("Elapsed time: " + str(time.time() - t0))
else:
    raise Exception("Argument at position 2 must be either .tflite or .pt")
