import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import torch
import os

def save_and_convert_model(model_name, model, dummy_input, input_names, output_names):
    torch.save(model.state_dict(), './trained_models/' + str(model_name) + "/" + str(model_name) + '.pt')

    if not os.path.exists('./trained_models/' + str(model_name)):
        os.mkdir('./trained_models/' + str(model_name))
    path = './trained_models/' + str(model_name) + "/" + str(model_name)
    torch.onnx.export(model, dummy_input, path + '.onnx', verbose=True, input_names=input_names,
                      output_names=output_names)
    onnx_model = onnx.load(path + ".onnx")
    onnx.checker.check_model(onnx_model)

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(path + "/")

    tflite_model_path = path + ".tflite"

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)