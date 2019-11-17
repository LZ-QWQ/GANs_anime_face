from model import DCGAN
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with tf.Session() as sess:
    dcgan=DCGAN(gpu_nums=1)
    dcgan.build_model_and_loss()    
    dcgan.add_optimizer()   
    dcgan.train(sess,epochs=250,batch_size=128,data_dir='anime_H_style')