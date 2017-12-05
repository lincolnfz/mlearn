# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:37:28 2017

@author: linco
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
import random
from sklearn import preprocessing
import sklearn

# 通过rcParams设置全局横纵轴字体大小
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

np.random.seed(42)

n = 100
# x轴的采样点
x = np.linspace(0, 1000, n)

# 通过下面曲线加上噪声生成数据，所以拟合模型就用y了……
y = 2*np.sin(x) + 0.3*x**2
#y = 13.57 + 5.3*x + x**2 * 5
#y = 12.57 + 5.3*x
y_data = y +  np.random.normal(scale=0.1, size=n)

scaler = sklearn.preprocessing.MinMaxScaler()
x = scaler.fit_transform(x.reshape(-1,1))
y_data = scaler.fit_transform(x.reshape(-1,1))
