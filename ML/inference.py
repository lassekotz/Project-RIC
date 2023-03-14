import numpy as np
#import tflite_runtime.interpreter as tflite
import torch
import torchvision.models as models
from model import CNN, LinearModel, test
from torch import nn
from preprocessing import ImagesDataset, generate_transforms, generate_dataloader
import sys
import time

def load_model_tflite(model_name):
    # TODO: Should output a model similar to load_model()
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
    if model_name == 'LinearModel':
        model = LinearModel()
    elif model_name == 'CNN':
        model = CNN()
    elif model_name == 'VGG':
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)
    else:
        raise Exception("Model " + model_name + " not available in ./trained_models/")
    print('./trained_models/' + model_name + "/" + model_name + ".pt")
    model.load_state_dict(torch.load("./trained_models/" + model_name + "/" + model_name + ".pt"))

    return model


if __name__ == '__main__':
    #model_name = str(sys.argv[1])
    #image_path = str(sys.argv[2])
    model_name = 'VGG.pt'
    image_path = './Data/BigDataset'

    model_name, extension = model_name.split('.')
    extension = '.' + extension
    all_transforms, no_transform, current_transform = generate_transforms(image_path)
    dataset = ImagesDataset(image_path, no_transform)
    batch_size = 1
    train_loader, val_loader, test_loader = generate_dataloader(dataset, batch_size, [0.01, 0.01, .98])#TODO: FIX THAT TEST_LOADER DOES NOT CONTAIN IMAGES THAT MODEL HAS SEEN

    if extension == '.pt':
        model = load_model(model_name)
    elif extension == '.tflite':
        interpreter = load_model_tflite(model_name)
    else:
        raise Exception("Argument at position 2 must be either .tflite or .pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    t0 = time.time()
    test(test_loader, model, device)
    print("Elapsed time: " + str(time.time() - t0))