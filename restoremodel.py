# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
saver = tf.train.import_meta_graph('./mnist-rnn-check/mnist_rnn_data.meta')
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./mnist-rnn-check'))
    graph = tf.get_default_graph()
    batch_size = graph.get_tensor_by_name('Placeholder:0')
    input_x = graph.get_tensor_by_name('Placeholder_1:0')
    keep_out = graph.get_tensor_by_name('Placeholder_3:0')
    #keep_out = graph.get_operation_by_name('Placeholder_3:0')
    #print(batch_size)
    out_op = graph.get_tensor_by_name('Softmax:0')
    imgs = mnist.test.images
    labs = mnist.test.labels
    item = imgs[0].reshape(1,-1)
    lab = labs[0]
    #print(item.shape)
    out = sess.run(out_op, feed_dict={input_x: item, keep_out:1.0, batch_size:1} )
    print(out[0])
    print(lab)