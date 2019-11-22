from flask import Flask,Response,send_file,render_template
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import os
import PIL.Image
from io import BytesIO
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

app=Flask('QAQ')
api = Api(app)

QAQ_path='model\\QAQ.pkl'
QwQ_path='model\\QwQ.pkl'
OvO_path='model\\OvO.tflite'
blue_hair_dir=np.load('model\\blue_hair.npy')
#DCGAN模型运行准备
interpreter = tf.lite.Interpreter(model_path=OvO_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']

class QAQ(Resource):
    def get(self):  
        parser = reqparse.RequestParser()
        parser.add_argument('truncation',type=float)
        parser.add_argument('blue_hair',type=float)
        args = parser.parse_args()
        truncation=args['truncation']
        blue_hair=args['blue_hair']
        if truncation is None:
            truncation=0.7
        if blue_hair is None:
            blue_hair=0.0
        print(blue_hair)
        if tf.get_default_session() is None:
            tflib.init_tf()
            Gs = pickle.load(open(QAQ_path, 'rb'))
        #这地方有个比较麻烦的问题，不能将Gs移到外面提前加载，好像跟session的行为有点关系，要是有办法不妨告诉我？？
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])
        latents=latents+blue_hair*blue_hair_dir
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
        if tf.get_default_session() is not None:
            tf.get_default_session().close()
            tf.get_default_session()._default_session.__exit__(None, None, None)
            #这玩意可以避免报一个错误，close()不知道有没有必要
        byte_io = BytesIO()
        PIL.Image.fromarray(images[0], 'RGB').save(byte_io,'png')      
        byte_io.seek(0)
        
        return send_file(byte_io, mimetype='image/png')

class QwQ(Resource):
    def get(self):  
        parser = reqparse.RequestParser()
        parser.add_argument('truncation',type=float)        
        args = parser.parse_args()
        truncation=args['truncation']
        
        if truncation is None:
            truncation=0.7

        if tf.get_default_session() is None:
            tflib.init_tf()
            Gs = pickle.load(open(QwQ_path, 'rb'))
        #这地方有个比较麻烦的问题，不能将Gs移到外面提前加载，好像跟session的行为有点关系，要是有办法不妨告诉我？？
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
        if tf.get_default_session() is not None:
            tf.get_default_session().close()
            tf.get_default_session()._default_session.__exit__(None, None, None)
            #这玩意可以避免报一个错误，close()不知道有没有必要
        byte_io = BytesIO()
        PIL.Image.fromarray(images[0], 'RGB').save(byte_io,'png')      
        byte_io.seek(0)
        
        return send_file(byte_io, mimetype='image/png')

class OvO(Resource):
    def get(self):
        input_=np.array(np.random.uniform(-1,1,size=input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_)
        interpreter.invoke()
        image = interpreter.get_tensor(output_details[0]['index'])
        byte_io = BytesIO()
        image=(image[0]+1)*127.5
        PIL.Image.fromarray(image.astype('uint8')).save(byte_io,'png')  
        byte_io.seek(0)
        
        return send_file(byte_io, mimetype='image/png')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/favicon.ico')
def get_fav():
    return send_file('static/favicon.ico')

api.add_resource(QAQ, '/QAQ')
api.add_resource(QwQ, '/QwQ')
api.add_resource(OvO, '/OvO')

if __name__ == '__main__':

    app.run(debug=False,host="0.0.0.0")