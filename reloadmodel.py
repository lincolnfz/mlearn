# -*- coding: UTF-8 -*-


import tensorflow.examples.tutorials.mnist.input_data
import input_data
from tensorflow.python.platform import gfile
import tensorflow as tf

pb_file_path = './mnist_model/'

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
#saver = tf.train.Saver()
saver = tf.train.import_meta_graph(pb_file_path + "model.ckpt.meta")

#sess = tf.Session()





with tf.Session() as sess:
    with gfile.FastGFile(pb_file_path+'model-nt.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') # 导入计算图
    saver.restore( sess, pb_file_path + 'model.ckpt' )
    graph = tf.get_default_graph()
    imgval = graph.get_tensor_by_name('Placeholder:0')
    keep_prob = graph.get_tensor_by_name('Placeholder_2:0')

    out = graph.get_tensor_by_name('prediction:0')
    for img, lab in zip(mnist.test.images, mnist.test.labels):
        feed_dict = { imgval: img.reshape(1,784) , keep_prob: 1.0 }
        #print(sess.run( out, feed_dict = feed_dict))
        #print(mnist.test.labels[0])
        pout =  sess.run( out, feed_dict=feed_dict )
        #correct_prediction = tf.equal(tf.argmax(pout, 1), tf.argmax(lab, 1))
        print(  pout , lab )
    

# 需要有一个初始化的过程    
#sess.run(tf.global_variables_initializer())


x = tf.placeholder(tf.float32, shape=[None, 784])

y_ = tf.placeholder("float", shape=[None,10])

