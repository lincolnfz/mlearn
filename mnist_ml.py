# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:31:02 2017

@author: linco
"""

# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
# https://www.tensorflow.org/get_started/mnist/beginners

import input_data
import tensorflow.examples.tutorials.mnist.input_data
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''开始创建softmax模型'''
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

'''训练模型'''
y_ = tf.placeholder("float", shape=[None,10])

'''下面是两段的损失函数的表达'''
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#cross_entropy = tf.reduce_mean(
#      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))