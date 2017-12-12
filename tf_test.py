# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:03:40 2017

@author: linco
"""
import math
import numpy as np
import tensorflow as tf

def transpi(ang):
    return ang * math.pi / 180.0

def mycos(ang):
    angpi = ang * math.pi / 180.0
    return math.cos(angpi)

def mysin(ang):
    angpi = ang * math.pi / 180.0
    return math.sin(angpi)

#print(mycos(180))

#print(mysin(270))

mat = np.array([[0,1],[1,1],[1,0]])
#print(mat.T)


p=[0.5, 0.25, 0.125, 0.125]
sum = 0;
for i in p:
    sum = sum + ( 0 - (i * np.log2(i)) )
    
#print(sum)
    
#tensorflow test   

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
#print(node1, node2)
sess = tf.Session()
print(sess.run([node1, node2]))

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#fixW = tf.assign(W, [-1.])
#fixb = tf.assign(b, [1.])
#sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(10000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
  
print(sess.run([W, b]))
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))