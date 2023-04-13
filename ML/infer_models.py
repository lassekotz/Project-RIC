import onnx
import onnxruntime as ort
from model import test
import torch
from torchvision import models
import numpy as np
from torch import nn
from preprocessing import generate_dataloader, ImagesDataset, generate_transforms

def pytorch_inf():
    model = models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 1)
    model.load_state_dict(torch.load("./trained_models/MobileNetV2/MobileNetV2.pt"))
    model.eval()
    preds_and_labels_pt = test(test_loader, model, device, write=True)

    return preds_and_labels_pt

def onnx_inf():
    onnx_model = onnx.load("./trained_models/MobileNetV2/MobileNetV2.onnx")
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession("./trained_models/MobileNetV2/MobileNetV2.onnx")
    for x, y in test_loader:
        outputs = ort_sess.run(None, {'input_1': np.array(x*255)})
        print(outputs)
        #pred, actual = classes

def tflite_inf():
    pass

def edgetpu_tflite_inf():
    pass

def openvino_inf():
    pass

if __name__ == '__main__':
    device = "cpu"
    H, W = 128, 128
    image_path = './Data/BigDataset'
    batch_size = 32
    all_transforms, no_transform, current_transform = generate_transforms(image_path, H, W)
    dataset = ImagesDataset(image_path, no_transform)
    _, _, test_loader = generate_dataloader(dataset, batch_size, [.8, .19, .01])
    preds_and_labels_pt = pytorch_inf()
    #onnx_inf()