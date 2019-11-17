import tensorflow as tf
import glob
import numpy as np
import cv2
import os
import time
from PIL import Image
from ops import *
#很奇怪的就是用tf.keras.layers结果不一样？？
class DCGAN():
    def __init__(self,gpu_nums=2,log_dir='log_TB'):

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #这真的是无奈之举
        #tf.keras.layers.BatchNormalization 不会自动将 update_ops 添加到tf.GraphKeys.UPDATE_OPS 这个collection中
        #https://blog.csdn.net/u014061630/article/details/85104491
        self.g_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='g_bn1')#这个设置是DCGAN的
        self.g_bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='g_bn2')
        self.g_bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='g_bn3')
        self.g_bn4 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='g_bn4')

        self.d_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='d_bn1')
        self.d_bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='d_bn2')
        self.d_bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='d_bn3')

        #self.g_bn1 = batch_norm(name='g_bn1')
        #self.g_bn2 = batch_norm(name='g_bn2')
        #self.g_bn3 = batch_norm(name='g_bn3')
        #self.g_bn4 = batch_norm(name='g_bn4')
        #self.d_bn1 = batch_norm(name='d_bn1')
        #self.d_bn2 = batch_norm(name='d_bn2')
        #self.d_bn3 = batch_norm(name='d_bn3')

        self.gpu_nums=gpu_nums
        self.log_dir=log_dir

        self.z_dim=100#输入维度 每个维度[-1,1]  我觉得要大一点才行

        self.gf_dim=64#生成器 倒数二层 filters
        self.output_dim=3#生成器 倒数一层 filters
        self.df_dim=64#判别器 第一层 filters
        self.output_size=(96,96)#其实我很想弄大点
        
        #生成器网络 我就瞎搭了吧还是      
        #Conv2DTranspose好像有点不对劲 用generator_2和sampler_2 tf官网好像，，Conv2DTranspose没什么问题啊？？
        #记录一下，用Conv2DTranspose训练就会炸成灰，，，  
        #修改下 真正的错误是tf.keras.layers.BatchNormalization 
        #不会自动将 update_ops 添加到tf.GraphKeys.UPDATE_OPS 这个collection中
        
        self.generator_list=[]          
        self.generator_list.append(tf.keras.layers.Dense(self.output_size[0]//16*self.output_size[1]//16*self.gf_dim*8,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),activation=None,use_bias=True,
                                    bias_initializer=tf.constant_initializer(0.0),name='g_d'))
        self.generator_list.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='g_bn1'))
        self.generator_list.append(tf.keras.layers.Conv2DTranspose(self.gf_dim*4,(5,5),(2,2),padding='same',activation=None,use_bias=True,
                                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                                                    bias_initializer=tf.constant_initializer(0.0),name='g_c1'))
        self.generator_list.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='g_bn2'))
        self.generator_list.append(tf.keras.layers.Conv2DTranspose(self.gf_dim*2,(5,5),(2,2),padding='same',activation=None,use_bias=True,
                                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                                                    bias_initializer=tf.constant_initializer(0.0),name='g_c2'))
        self.generator_list.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='g_bn3'))
        self.generator_list.append(tf.keras.layers.Conv2DTranspose(self.gf_dim,(5,5),(2,2),padding='same',activation=None,use_bias=True,
                                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                                                    bias_initializer=tf.constant_initializer(0.0),name='g_c3'))
        self.generator_list.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='g_bn4'))
        self.generator_list.append(tf.keras.layers.Conv2DTranspose(self.output_dim,(5,5),(2,2),padding='same',activation=None,use_bias=True,
                                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                                                    bias_initializer=tf.constant_initializer(0.0),name='g_c4'))

        #判别器网络
        
        self.discriminator_list=[]           
        self.discriminator_list.append(tf.keras.layers.Conv2D(self.df_dim,(5,5),(2,2),padding='same',activation=None,use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                                bias_initializer=tf.constant_initializer(0.0),name='d_c1'))
        self.discriminator_list.append(tf.keras.layers.Conv2D(self.df_dim*2,(5,5),(2,2),padding='same',activation=None,use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                                bias_initializer=tf.constant_initializer(0.0),name='d_c2'))
        self.discriminator_list.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='d_bn1'))
        self.discriminator_list.append(tf.keras.layers.Conv2D(self.df_dim*4,(5,5),(2,2),padding='same',activation=None,use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                                bias_initializer=tf.constant_initializer(0.0),name='d_c3'))
        self.discriminator_list.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='d_bn2'))
        self.discriminator_list.append(tf.keras.layers.Conv2D(self.df_dim*8,(5,5),(2,2),padding='same',activation=None,use_bias=True,
                                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                                bias_initializer=tf.constant_initializer(0.0),name='d_c4'))
        self.discriminator_list.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum = 0.9,name='d_bn3'))
        self.discriminator_list.append(tf.keras.layers.Dense(1,activation=None,use_bias=True,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    bias_initializer=tf.constant_initializer(0.0),name='d_d'))

    def build_model_and_loss(self):
        with tf.variable_scope('QAQ') as scope:
            self.z=tf.placeholder(tf.float32,(None,self.z_dim),name='g_input')
            self.images=tf.placeholder(tf.float32,(None,self.output_size[0],self.output_size[1],self.output_dim),name='d_input')
            
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.tower_G_z=[]
        self.tower_G_sample=[]
        self.tower_D_image=[]
        self.tower_D_z=[]

        with tf.device('/cpu:0'):#我的两个GPU不能均分啊其实！！！
            tower_z=tf.split(self.z,num_or_size_splits=self.gpu_nums, axis=0)
            tower_image=tf.split(self.images,num_or_size_splits=self.gpu_nums, axis=0)

        self.tower_d_loss=[]
        self.tower_g_loss=[]
        self.d_loss=0
        self.g_loss=0
        self.d_loss_real=0
        self.d_loss_fake=0

        g_list = tf.global_variables()
        var_list = tf.trainable_variables()

        gpus = ['/gpu:{}'.format(i) for i in range(self.gpu_nums)]
        for i in range(self.gpu_nums):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/cpu:0', worker_device=gpus[i])):                
            #其实这个设置我也不是很理解，但是有例子是这样做的，不记得网址了，Tacotron2源码就是这样的
                self.tower_G_z.append(self.generator(tower_z[i]))
                self.tower_G_sample=(self.sampler(tower_z[i]))
                self.tower_D_image.append(self.discriminator(tower_image[i]))
                self.tower_D_z.append(self.discriminator(self.tower_G_z[i]))   

                #判别器出来的是个元组的列表，第一个sigmoid第二个没有,不会是这里害死人吧？？？
                tower_d_loss_real=tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.tower_D_image[i][1],labels=tf.ones_like(self.tower_D_image[i][0])))
                tower_d_loss_fake=tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.tower_D_z[i][1],labels=tf.zeros_like(self.tower_D_z[i][0])))
                tower_g_loss=tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.tower_D_z[i][1],labels=tf.ones_like(self.tower_D_z[i][0])))

                tower_d_loss=tower_d_loss_real+tower_d_loss_fake
                self.tower_d_loss.append(tower_d_loss)
                self.tower_g_loss.append(tower_g_loss)
                self.d_loss+=tower_d_loss
                self.g_loss+=tower_g_loss
                self.d_loss_fake+=tower_d_loss_fake
                self.d_loss_real+=tower_d_loss_real

        self.QAQ_output=tf.concat(self.tower_G_sample,0,name='QAQ_output')#最终的输出！

        #记录下tensorboard
        #G_sum = tf.summary.image("G", tf.concat(self.tower_G_z,0))这样保存也太大了吧？？
        d_sigmoid_z_sum = tf.summary.histogram("d_sigmoid_z", tf.math.sigmoid(tf.concat([temp[0] for temp in self.tower_D_z],0)))
        d_sigmoid_image_sum = tf.summary.histogram("d_sigmoid_image", tf.math.sigmoid(tf.concat([temp[0] for temp in self.tower_D_image],0)))
        #上面这两个是因为emmm，我为什么判别器要返回两个值啊？？？,上面这样应该就行了
        d_loss_sum=tf.summary.scalar('d_loss',self.d_loss)
        g_loss_sum=tf.summary.scalar('g_loss',self.g_loss)
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_sum=tf.summary.merge([d_loss_sum,d_loss_real_sum,d_loss_fake_sum,d_sigmoid_z_sum,d_sigmoid_image_sum])
        self.g_sum=tf.summary.merge([g_loss_sum])
        

        #说是BN可能保存不进去
        g_list = tf.global_variables()
        var_list = tf.trainable_variables()
        temp_list=[var for var in g_list if ('g_bn' in var.name or 'd_bn' in var.name or 'global_step' in var.name) and var not in var_list]
        var_list+=temp_list
        self.saver=tf.train.Saver(var_list=var_list,max_to_keep=100)
        print('参数总量~(trainable):{:.3f}.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    def add_optimizer(self):
        #根据gpu_nums设置梯度计算~
        #就先这样设置吧~
        with tf.device('/cpu:0'):
            with tf.variable_scope('optimizer') as scope:
                d_lr=tf.train.exponential_decay(learning_rate=0.0002,global_step=self.global_step,
                        decay_steps=30000,decay_rate=0.5,staircase=False,name='d_lr_decay')
                g_lr=tf.train.exponential_decay(learning_rate=0.0002,global_step=self.global_step,
                        decay_steps=30000,decay_rate=0.5,staircase=False,name='g_lr_decay')
                d_optim = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=0.5)
                g_optim = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=0.5)


        #多GPU设置 参考 Tacotron2
        tower_d_gradients=[]
        tower_g_gradients=[]
        gpus = ['/gpu:{}'.format(i) for i in range(self.gpu_nums)]
        for i in range(self.gpu_nums):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/cpu:0', worker_device=gpus[i])):
                with tf.variable_scope('optimizer'):
                    d_gradient=d_optim.compute_gradients(self.tower_d_loss[i])
                    g_gradient=g_optim.compute_gradients(self.tower_g_loss[i])
                    tower_d_gradients.append(d_gradient)
                    tower_g_gradients.append(g_gradient)
        
        with tf.device('/cpu:0'):
            avg_d_grads=[]
            avg_g_grads=[]
            d_variables=[]
            g_variables=[]

            #判别器的
            for grad_and_vars in zip(*tower_d_gradients):
			# each_grads_vars = ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                temp=grad_and_vars[0][1]
                if 'g_' in temp.name:
                    continue #要不判别器会把生成器也更新了！！！我靠！！
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
					#添加一个维度用了存储每个tower的梯度随后以此维度求平均
                    grads.append(expanded_g)
				#Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                variable = grad_and_vars[0][1]
                avg_d_grads.append(grad)
                d_variables.append(variable)

            #生成器的
            for grad_and_vars in zip(*tower_g_gradients):
			# each_grads_vars = ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                temp=grad_and_vars[0][1]
                if 'd_' in temp.name:
                    continue #要不生成器会把判别器也更新了！！！我靠！！
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
					#添加一个维度用了存储每个tower的梯度随后以此维度求平均
                    grads.append(expanded_g)
				#Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                variable = grad_and_vars[0][1]
                avg_g_grads.append(grad)
                g_variables.append(variable)
        
        #这是一个害死人的地方，，，
        #https://blog.csdn.net/u014061630/article/details/85104491
        #有点想骂人。。。。debug一辈子。。。
        ops = tf.get_default_graph().get_operations()
        update_ops = [x for x in ops if ("AssignMovingAvg" in x.name and x.type=="AssignSubVariableOp")]
        for temp in update_ops:
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, temp)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        g_ops=[]

        for temp in update_ops:
            if 'generator' in temp.name or 'discriminator_1' in temp.name:
                g_ops.append(temp)

        with tf.control_dependencies(update_ops):
            self.d_optim = d_optim.apply_gradients(zip(avg_d_grads, d_variables),global_step=self.global_step)
        with tf.control_dependencies(g_ops):
            self.g_optim = g_optim.apply_gradients(zip(avg_g_grads, g_variables))

    def train(self,sess,epochs=200,batch_size=256,data_dir='anime_H',path_checkpoint='model_save',image_save_path='image_save',
        path_checkpoint_prefixname='model_save\\DCGAN',sampler_epochs=1,save_epochs=3):
        
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)

        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

        self.sess=sess
        init=tf.global_variables_initializer()        
        self.sess.run(init)

        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        #以下为获取数据的乱操作(⊙﹏⊙)
        data_list=glob.glob(data_dir+'\\*.*')
        data_list=[data.split('\\')[-1] for data in data_list]
        pos=0
        data_len=len(data_list)
        batch_size=batch_size
        steps=data_len//batch_size+1#这样算step~
        self.load(path_checkpoint)

        for epoch in range(epochs):
            np.random.shuffle(data_list)#我也不知道这有没有用会不会有影响~
            for step in range(steps):
                start_time = time.time()
                #index_list=np.random.randint(len(data_list),size=batch_size)

                #还是遍历吧 steps指的是一个epoch走多少个batch
                if pos+batch_size>data_len:
                    data=[data_list[np.mod(i,data_len)] for i in range(pos,pos+batch_size)]
                else:
                    data=[data_list[i] for i in range(pos,pos+batch_size)]
                pos=np.mod(pos+batch_size,data_len)
                images_input=self.get_image(data,data_dir)
                zz=np.random.uniform(-1,1,size=(batch_size,self.z_dim))
                zz=zz.astype(np.float32)
                #这就只能是[-1,1)啊啊啊

                #DCGAN源码是 每步更新一次判别器和两次生成器
                _,global_step,d_sum=self.sess.run([self.d_optim,self.global_step,self.d_sum],feed_dict={self.z:zz,self.images:images_input})
                _=self.sess.run(self.g_optim,feed_dict={self.z:zz})
                _,g_sum=self.sess.run([self.g_optim,self.g_sum],feed_dict={self.z:zz})

                #其实这几个可以写进sess.run里
                errD_fake = self.d_loss_fake.eval({self.z:zz,self.images:images_input})
                errD_real = self.d_loss_real.eval({self.z:zz,self.images:images_input})
                errG = self.g_loss.eval({self.z:zz})
                print('{:d}/{:d}epoch  {:d}/{:d} step {:.4f}sec/step {:d} glob_steps d_loss: {:.4f} g_loss: {:.4f}  real:{:.4f} fake:{:.4f} '.format((epoch+1),epochs,
                      (step+1),steps,time.time()-start_time,global_step,errD_fake+errD_real,errG,errD_real,errD_fake),end='\r')
                self.writer.add_summary(d_sum,global_step)
                self.writer.add_summary(g_sum,global_step)
        
            if np.mod(epoch+1,sampler_epochs)==0:
                sample_size=batch_size#要可开方~,这是运行在单卡上的~
                sample_z=np.random.uniform(-1,1,size=(sample_size,self.z_dim))
                sample_z=sample_z.astype(np.float32)                
                index_list=np.random.randint(len(data_list),size=sample_size)
                data=[data_list[i] for i in index_list]
                sample_input=self.get_image(data,data_dir)
                samples,d_loss,g_loss = self.sess.run([self.QAQ_output,self.d_loss,self.g_loss],
                                                      feed_dict={self.z:sample_z,self.images:sample_input})
                #这个d_loss,g_loss是用一个随机噪声弄出来的
                self.save_image(samples,os.path.join(image_save_path,str(global_step)+'.png'))
                print()
                print("[Sample] d_loss: {:.4f}, g_loss: {:.4f}" .format(d_loss, g_loss)) 

            if np.mod(epoch+1,save_epochs)==0:
                self.saver.save(self.sess,save_path=path_checkpoint_prefixname,global_step=self.global_step)

    def generator(self,z):
        with tf.variable_scope('generator',reuse=tf.AUTO_REUSE)as scope:
            g=z
            #这里的顺序是因为tensorflow lite转换问题（bug吧？）决定改的，
            #貌似和https://github.com/mattya/chainer-DCGAN这个一样了~应该可行~
            for i,layer in enumerate(self.generator_list):
                if i==1:
                    g=layer(g,training=True)
                    g=tf.nn.relu(g)
                    g=tf.reshape(g,(-1,self.output_size[0]//16,self.output_size[1]//16,self.gf_dim*8))
                elif i in [1,3,5,7]:
                    g=layer(g,training=True)
                    g=tf.nn.relu(g)
                else:
                    g=layer(g)
        return tf.nn.tanh(g)    

    def sampler(self,z):
        with tf.variable_scope('generator',reuse=tf.AUTO_REUSE)as scope:
            g=z
            #这里的顺序是因为tensorflow lite转换问题（bug吧？）决定改的，
            #貌似和https://github.com/mattya/chainer-DCGAN这个一样了~应该可行~
            for i,layer in enumerate(self.generator_list):
                if i==1:
                    g=layer(g,training=False)
                    g=tf.nn.relu(g)
                    g=tf.reshape(g,(-1,self.output_size[0]//16,self.output_size[1]//16,self.gf_dim*8))
                elif i in [3,5,7]:
                    g=layer(g,training=False)
                    g=tf.nn.relu(g)
                else:
                    g=layer(g)
        return tf.nn.tanh(g)   

    def discriminator(self,image):
        with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE) as scope:
            d=image
            for i,layer in enumerate(self.discriminator_list):
                if i==0:
                    d=layer(d)
                    d=tf.nn.leaky_relu(d,alpha=0.2)
                elif i in [2,4,6]:
                    d=layer(d,training=True)
                    d=tf.nn.leaky_relu(d,alpha=0.2)
                elif i==7:
                    d=tf.reshape(d,(-1,self.output_size[0]//16*self.output_size[1]//16*self.df_dim*8))
                    d=layer(d)
                else:
                    d=layer(d)
        return tf.nn.sigmoid(d),d

    def load(self,path_checkpoint):
        ckpt=tf.train.get_checkpoint_state(path_checkpoint)

        if ckpt != None:
            path=ckpt.model_checkpoint_path
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,path)
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")

    def get_image(self,image_list,data_dir):
        images_input=np.zeros((len(image_list),self.output_size[0],self.output_size[1],self.output_dim),dtype=np.float32)

        for i in range(len(image_list)):
            image=cv2.imread(os.path.join(data_dir,image_list[i]))
            image=image[...,::-1]#要转一下BGR，据说这个写法快
            images_input[i]=cv2.resize(image,(self.output_size[0],self.output_size[1]),interpolation=cv2.INTER_AREA)
            #防止图片不是64，inter的设置据说缩小用这个AREA，因为取图是96故选这个先！！
            #images_input[i]=image#emmm一样就不转了吧先            
        images_input=images_input.astype(np.float32)/127.5-1#归一化(⊙﹏⊙)！！！
        return images_input

    def save_image(self,sample,filename):
        #这里要重整一下图片,sample第一维总数必须可开方
        temp_len=int(np.sqrt(sample.shape[0]))
        img=np.zeros(shape=(temp_len*sample.shape[1],temp_len*sample.shape[2],sample.shape[3]),dtype=np.float32)
        for i in range(int(np.sqrt(sample.shape[0]))):
            for j in range(int(np.sqrt(sample.shape[0]))):
                img[i*self.output_size[0]:(i+1)*self.output_size[0],j*self.output_size[1]:(j+1)*self.output_size[1],...]=sample[i*temp_len+j,...]
        img=(img+1)*127.5
        img=Image.fromarray(img.astype('uint8'))
        img.save(filename)
        #我实在找不到别的方法了！各种不能正常保存，日！！

    def load_and_convert(self,sess,save_filename,ckpt_name='LZ'):
        #声明类后直接用 务必设置gpu=1
        #参考https://stackoverflow.com/questions/56110234/dimensions-must-match-error-in-tflite-conversion-with-toco
        #参考https://zhuanlan.zhihu.com/p/66346329
        if self.gpu_nums!=1:
            print('最好别这样QAQ')
            raise ValueError('不要啊~')
        path='convert2lite_ckpt'
        self.build_model_and_loss()
        self.add_optimizer()
        sess.run(tf.global_variables_initializer())
        self.saver.restore(sess,os.path.join(path,ckpt_name))       
        converter = tf.lite.TFLiteConverter.from_session(sess, [self.z], [self.QAQ_output])
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(save_filename, "wb") as f:
            f.write(tflite_model)