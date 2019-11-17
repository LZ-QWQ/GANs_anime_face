import numpy as np
import tensorflow as tf
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#这说明我的tensorflow lite 没问题啊
#用这个来输出吧~
#运行这个文件就行啦~要有QAQ.tflite哦~先运行convert2lite
# Load TFLite model and allocate tensors.

total=10#多少次~
nums=15#每次输出多少张图片

interpreter = tf.lite.Interpreter(model_path="QAQ.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
img_end=np.zeros(shape=(total*output_shape[1],nums*output_shape[2],output_shape[3]),dtype=np.float32)
for k in range(total):

    a=np.array(np.random.uniform(-1,1,size=input_shape), dtype=np.float32)
    b=np.array(np.random.uniform(-1,1,size=input_shape), dtype=np.float32)
    c=(b-a)/(nums-1)
    img_list=[]
    for i in range(nums):
        temp=a+c*i
        interpreter.set_tensor(input_details[0]['index'], temp)
        interpreter.invoke()
        img = interpreter.get_tensor(output_details[0]['index'])
        img_list.append(img)
    all_img=np.squeeze(np.concatenate(img_list,axis=2))
    img_end[k*output_shape[1]:(k+1)*output_shape[1],...]=all_img



img_end=(img_end+1)*127.5
img_end=Image.fromarray(img_end.astype('uint8'))
img_end.save('QAQ.png')