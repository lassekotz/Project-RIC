import pathlib

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

    torch.onnx.export(model, dummy_input, path + '.onnx', verbose=False, input_names=input_names,
                      output_names=output_names)
    onnx_model = onnx.load(path + ".onnx")
    onnx.checker.check_model(onnx_model)

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(path + "/")

    tflite_model_path = path + ".tflite"
    tflite_model_quant_path = path + "_quant.tflite"

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite_model = converter.convert()

    def representative_data_gen():
        #dataset_list = tf.data.Dataset.list_files('C:/Users/jonalm3/Project-RIC/ML/Data/BigDataset/images' + "/*/*")
        for i in range(500):
            #image = next(iter(dataset_list))
            image = tf.io.read_file("C:/Users/jonalm3/Project-RIC/ML/Data/BigDataset/images/" + str(i) + ".jpg")
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [128, 128])
            image = tf.cast(image / 255., tf.float32)
            image = tf.expand_dims(image, 0)
            yield [image]
        #for input_value in tf.data.Dataset.from_tensor_slices(dummy_input.cpu().numpy()).batch(1).take(2):
            #yield [input_value]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    print(interpreter.get_input_details())
    input_type = interpreter.get_input_details()[0]['dtype']
    output_type = interpreter.get_output_details()[0]['dtype']
    print("input: ", input_type)
    print("output: ", output_type)

    # TODO: modify quantization to full int8

    # Save the model

    tflite_model_path = pathlib.Path(tflite_model_path)
    tflite_model_path.write_bytes(tflite_model)
    tflite_model_quant_path = pathlib.Path(tflite_model_quant_path)
    tflite_model_quant_path.write_bytes(tflite_model_quant)