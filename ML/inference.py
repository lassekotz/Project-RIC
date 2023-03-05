import numpy as np
import time
import tflite_runtime.interpreter as tflite
import sys
import torchvision.models as models
from model import CNN, LinearModel
from torch import nn

#model_name = str(sys.argv[1])
model_name = 'vgg16.pt'
model_name, extension = model_name.split('.')
extension = '.' + extension

def load_model_tflite(model_name):
    print("inference running on: " + model_name)
    interpreter = tflite.Interpreter(model_path="trained_models/" + model_name + "/" + model_name + ".tflite")
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

def load_model(model_name):
    if model_name == 'linear':
        model = LinearModel()
    elif model_name == 'CNN':
        model = CNN()
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)
    else:
        raise Exception("Model " + model_name + " not available in ./trained_models/")
    model.load_state_dict('./trained_models/' + model_name + "/" + model_name + ".pt")


if extension == '.pt':
    model = load_model(model_name)
elif extension == '.tflite':
    interpreter = load_model_tflite(model_name)
else:
    raise Exception("Argument at position 2 must be either .tflite or .pt")

