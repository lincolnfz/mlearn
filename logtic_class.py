# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:30:19 2017

@author: linco
"""

# http://www.jianshu.com/p/d94e40e2df81
# logistic分类

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import colorConverter, ListedColormap
from matplotlib import cm

# Define the logistic function
def logistic(z): 
    return 1 / (1 + np.exp(-z))

# Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
def nn(x, w): 
    return logistic(x.dot(w.T))

# Define the neural network prediction function that only returns
#  1 or 0 depending on the predicted class
def nn_predict(x,w): 
    return np.around(nn(x,w))
    
# Define the cost function
def cost(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

# define the gradient function.
def gradient(w, x, t):
    return (nn(x, w) - t).T * x

# define the update function delta w which returns the 
#  delta w for each weight in a vector
def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t)

