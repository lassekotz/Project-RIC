import sys
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from torchvision import models
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import torch
from torch import nn
from model import test, get_model
import os

def view_label_distr(filepath, bins=10):
    with open(filepath) as f:
        lines = f.readlines()
        vals = []
        for line in lines:
            vals.append(float(line.replace('\n', '')))

    vals_np = np.array(vals)
    mean = np.mean(vals_np)
    plt.axvline(x=mean, color='r', label='axvline - full height')
    plt.hist(vals_np, bins=bins)

    plt.legend(['mean = %.2f' % mean, 'label distr.'])
    plt.title('Distribution of angles in dataset ' + str(filepath))
    plt.show()

    return None


def plot_error_distr(errors_list, bins=20):
    errors_np = np.array(errors_list)

    fix, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(errors_np, bins=60)
    ax1.set_title('Error distribution')
    ax1.set_xlim(-10, 10)
    ax2.plot(errors_list)
    ax2.set_title('Errors over time')
    plt.show()

    return None


def plot_pred_space(targets, preds, MAE):
    plt.scatter(preds, targets, .5)
    plt.plot([-30, 30], [-30, 30], 'r-')
    plt.title(f'Prediction space, MAE = ' + f'{MAE:.2f}')
    plt.legend(['Ideal', 'Predictions'])
    plt.xlabel('Predicted angle')
    plt.ylabel('Actual angle')
    plt.grid()
    plt.xticks()
    plt.yticks()
    plt.show()
    # TODO: fix this

    return None

def plot_pred_space_heatmap(targets, preds, MAE, bins = 200):
    heatmap, xedges, yedges = np.histogram2d(preds, targets, bins=200)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title(f'Prediction space, MAE = ' + f'{MAE:.2f}. No. of bins = ' + f'{bins:.0f}')
    plt.xlabel('Predicted angle')
    plt.ylabel('Actual angle')
    plt.colorbar()
    plt.grid()
    plt.xticks()
    plt.yticks()
    plt.show()


def plot_pred_target_distributions(targets, preds, bins=20):
    fix, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(targets, bins)
    ax1.set_title('Target distribution')
    ax2.hist(preds, bins)
    ax2.set_title('Prediction distribution')
    plt.show()


def plot_results(train_losses, val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['train losses', 'val losses'])
    plt.title('Training progress')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.grid()
    plt.show()

def fourier(errors_list):
    x = np.array(errors_list)
    fourier = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), .1)
    plt.xlabel('w')
    plt.ylabel('')
    plt.title('Fourier Transform')
    plt.plot(freq, abs(fourier)**2)
    plt.show()
    #print(y)

def compare_conversions():
    path = "trained_models/MobileNetV2/MobileNetV2"
    onnx_model = onnx.load(path + ".onnx")
    tflite_interpreter = tf.lite.Interpreter(model_path=path + ".tflite")

def visualize_feature_maps():
    model = get_model("mobilenet_v2")
    model_weights = []
    conv_layers = []
    nr_of_conv2d = 0
    model_children = list(model.children())
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            nr_of_conv2d += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        nr_of_conv2d += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    outputs = []
    names = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loader = torch.load("Data/Dataloaders/test_loader.pth")
    for layer in conv_layers:
        for (x, y) in test_loader:
            feature_map = layer(x)
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]

            fig, axs = plt.subplots(2)
            axs[0].imshow(gray_scale.detach().numpy())
            axs[1].imshow(x.squeeze(0).swapaxes(0, 2).swapaxes(0, 1))
            plt.show()
            #break
        #break



        #image = layer(image)
        #outputs.append(image)
        #names.append(str(layer))

    for feature_map in outputs:
        print(feature_map.shape)

    pass # TODO: implement plotting of feature maps in different error-ranges; MAE 0-mean, MAE mean-std1, MAE mean-std2




if __name__ == '__main__':
    '''
    datapath = './Results/MobileNetV2/test_results.txt'

    with open(datapath) as f:
        lines = f.readlines()
        targets = []
        preds = []
        for line in lines:
            line = line.replace('(', '').replace(')', '')
            line = tuple(map(float, line.split(', ')))
            targets.append(line[0])
            preds.append(line[1])

    MAE = 0
    errors_list = []
    for i in range(len(preds)):
        t = targets[i]
        p = preds[i]
        MAE += abs(t-p)
        errors_list.append(t-p)
    MAE = MAE/len(preds)

    plot_pred_space(targets, preds, MAE)
    plot_pred_target_distributions(targets, preds, bins=50)
    plot_pred_space_heatmap(targets, preds, MAE)
    plot_error_distr(errors_list)
    '''
    test_loader = torch.load("Data/Dataloaders/test_loader.pth")
    visualize_feature_maps()