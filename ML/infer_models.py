import onnx
import onnxruntime as ort
from model import test
import torch
from torchvision import models
import numpy as np
from torch import nn
from preprocessing import generate_dataloader, ImagesDataset, generate_transforms
import matplotlib.pyplot as plt
from openvino.runtime import Core

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
    preds_and_labels_onnx = []
    for x, y in test_loader:
        outputs = ort_sess.run(None, {'input_1': np.array(x)})
        pred = outputs[0][0][0].item()
        label = y.item()
        preds_and_labels_onnx.append((pred, label))

    return preds_and_labels_onnx

def tflite_inf():
    pass

def openvino_inf():
    ie = Core()
    model_xml = "trained_models/MobileNetV2/MobileNetV2.xml"
    model = ie.read_model(model=model_xml)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    preds_and_labels_openvino = []

    for x, y in test_loader:
        input_data = np.array(x)
        pred = compiled_model(input_data)[output_layer]
        preds_and_labels_openvino.append((pred.item(), y.item()))

    return preds_and_labels_openvino


def edgetpu_tflite_inf():
    pass

def compare_conversions(all_preds):
    color_dict = {
        'pt': 'r',
        'onnx': 'b',
        'openvino': 'g'
    }
    plt.plot([-30, 30], [-30, 30], 'r-')
    for key in all_preds.keys():
        predss, targetss = [], []
        for i in range(len(all_preds[key])):
            predss.append(all_preds[key][i][0])
            targetss.append(all_preds[key][i][1])
        plt.scatter(predss, targetss, .5, color_dict[key])
    plt.title(f'Prediction space, MAE = ')
    plt.legend(['ideal'] + [key for key in color_dict.keys()])

    plt.xlabel('Predicted angle')
    plt.ylabel('Actual angle')
    plt.grid()
    plt.xticks()
    plt.yticks()
    plt.show()


if __name__ == '__main__':
    device = "cpu"
    H, W = 128, 128
    image_path = './Data/BigDataset'
    batch_size = 32
    all_transforms, no_transform, current_transform = generate_transforms(image_path, H, W)
    dataset = ImagesDataset(image_path, no_transform)
    _, _, test_loader = generate_dataloader(dataset, batch_size, [.8, .19, .01])

    all_preds = {}
    preds_and_labels_pt = pytorch_inf()
    preds_and_labels_onnx = onnx_inf()
    preds_and_labels_openvino = openvino_inf()

    all_preds['pt'] = preds_and_labels_pt
    all_preds['onnx'] = preds_and_labels_onnx
    all_preds['openvino'] = preds_and_labels_openvino

    compare_conversions(all_preds)