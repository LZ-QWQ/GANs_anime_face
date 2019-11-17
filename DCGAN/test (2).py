import imageio
import scipy.misc
from PIL import Image
import numpy as np
test=Image.open('image_save\\1407.png')
test=np.asarray(test)
print(test.max())
test=test/127.5-1
img=Image.fromarray(test.astype('uint8'))
img.save('test.png')

test=(test+1)*127.5
img=Image.fromarray(test.astype('uint8'))
img.save('test_2.png')

    def build_model_and_loss_fuck(self):
        with tf.variable_scope('QAQ') as scope:
            self.z=tf.placeholder(tf.float32,(self.batch_size,self.z_dim),name='g_input')
            self.images=tf.placeholder(tf.float32,(self.batch_size,self.output_size[0],self.output_size[1],self.output_dim),name='d_input')
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.G_z=(self.generator_2(self.z))
        self.G_sample=(self.sampler_2(self.z))
        self.D_image,D_image_logits=(self.discriminator(self.images))
        self.D_z,D_z_logits=(self.discriminator(self.G_z))#emmm

        def loss_(x,y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=y)

        self.d_loss_real=tf.math.reduce_mean(loss_(D_image_logits,tf.ones_like(self.D_image)))
        self.d_loss_fake=tf.math.reduce_mean(loss_(D_z_logits,tf.zeros_like(self.D_z)))
        self.g_loss=tf.math.reduce_mean(loss_(D_z_logits,tf.ones_like(self.D_z)))

        self.d_loss=self.d_loss_real+self.d_loss_fake           

        #记录下tensorboard
        #G_sum = tf.summary.image("G", tf.concat([self.tower_G_z[0],self.tower_G_z[1]],0))这样保存也太大了吧？？
        d_sigmoid_z_sum = tf.summary.histogram("d_sigmoid_z", self.D_z)
        d_sigmoid_image_sum = tf.summary.histogram("d_sigmoid_image", self.D_image)
        d_loss_sum=tf.summary.scalar('d_loss',self.d_loss)
        g_loss_sum=tf.summary.scalar('g_loss',self.g_loss)
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_sum=tf.summary.merge([d_loss_sum,d_loss_real_sum,d_loss_fake_sum,d_sigmoid_z_sum,d_sigmoid_image_sum])
        self.g_sum=tf.summary.merge([g_loss_sum])

        g_list = tf.global_variables()
        var_list = tf.trainable_variables()
        temp_list=[var for var in g_list if ('g_bn' in var.name or 'd_bn' in var.name or 'global_step' in var.name) and var not in var_list]
        var_list+=temp_list
        self.saver=tf.train.Saver(var_list=var_list,max_to_keep=1)
        print('参数总量~(trainable):{:.3f}.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))