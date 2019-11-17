from model import DCGAN
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
with tf.Session() as sess:   

    graph_def_file = "test_save.pb"
    input_arrays = ["QAQ/g_input"]
    output_arrays = ["QAQ_output",'Tanh']

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    with open("converted_model.tflite", "wb") as f:
        f.write(tflite_model)