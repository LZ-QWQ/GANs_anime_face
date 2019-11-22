import tensorflow as tf
import numpy as np
import os

#我没想明白这玩意咋控制模长比较好。。。。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#参数变量初始化
W = tf.Variable(tf.random_normal(shape=[512,1],mean=0,stddev=1),name="weights")#变量权值
b = tf.Variable(0., name="bias")#线性函数常量，模型偏置
def combine_inputs(X):#输入值合并
    print ("function: combine_inputs")
    return tf.matmul(X, W) + b
def inference(X):#计算返回推断模型输出(数据X)
    print ("function: inference")
    return tf.sigmoid(combine_inputs(X))#调用概率分布函数
def loss(X, Y):#计算损失(训练数据X及期望输出Y)
    print ("function: loss")
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))#求平均值
def inputs():#读取或生成训练数据X及期望输出Y
    X=[]
    for i in range(0,7000):
        temp=np.load('G_latents(0.7)\\example-'+str(i)+'.npy')
        X.append(temp[0])
    X=np.array(X,dtype=np.float32)
    Y=np.load('tag_labels.npy')
    Y=Y.reshape([-1,1])
    Y=Y.astype(dtype=np.float32)
    return X,Y
def train(total_loss):#训练或调整模型参数(计算总损失)
    print ("function: train")
    learning_rate = 0.5
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
def evaluate(sess, X, Y):#评估训练模型
    print ("function: evaluate")
    predicted = tf.cast(inference(X) > 0.5, tf.float32)#样本输出大于0.5转换为正回答
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))#统计所有正确预测样本数，除以批次样本总数，得到正确预测百分比
#会话对象启动数据流图，搭建流程
with tf.Session() as sess:
    print ("Session: start")
    init = tf.global_variables_initializer()
    sess.run(init)
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    training_steps = 10000#实际训练迭代次数
    for step in range(training_steps):#实际训练闭环
        sess.run([train_op])
        if step % 10 == 0:#查看训练过程损失递减
            print(str(step)+ " loss: ", sess.run([total_loss]))
    print(str(training_steps) + " final loss: ", sess.run([total_loss]))
    temp=W.eval().reshape([1,-1])
    #temp=temp/np.sqrt(np.sum(np.square(temp)))
    np.save('vector',temp)