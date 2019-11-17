from model import DCGAN
import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#把cpkt那三个文件放在convert2lite_ckpt下，改名称!
with tf.Session() as sess:
    dcgan=DCGAN(gpu_nums=1)
    dcgan.load_and_convert(sess,'QAQ.tflite',ckpt_name='LZ')