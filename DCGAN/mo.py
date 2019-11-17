from model import DCGAN
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
with tf.Session() as sess:   
    saver = tf.train.import_meta_graph('temp\\' + 'DCGAN-5025.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    temp=graph._nodes_by_name
    
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    ckpt=tf.train.get_checkpoint_state('temp\\')
    path=ckpt.model_checkpoint_path
    saver.restore(sess, path)  # 恢复图并得到数据
    g_list = tf.global_variables()
    output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=['QAQ_output','Tanh'])  # 如果有多个输出节点，以逗号隔开
 
    with tf.gfile.GFile('test_save.pb', "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点