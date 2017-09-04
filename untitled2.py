# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:05:52 2017

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
y = 2*np.sin(x) + x**2 + x**4
#y = 13.57 + 5.3*x + x**2 * 5 + x**4 * 3.2
#y = 12.57 + 5.3*x
y_data = y +  np.random.normal(scale=9, size=n)

scaler = sklearn.preprocessing.MinMaxScaler()
x = scaler.fit_transform(x.reshape(-1,1))
y_data = scaler.fit_transform(y_data.reshape(-1,1))

# figure()指定图表名称
plt.figure('data')

# '.'标明画散点图，每个散点的形状是个圆
#plt.plot(x, y_data, '.')

# 画模型的图，plot函数默认画连线图
plt.figure('model')
#plt.plot(x, y)

# 两个图画一起
#plt.figure('data & model')

# 通过'k'指定线的颜色，lw指定线的宽度
# 第三个参数除了颜色也可以指定线形，比如'r--'表示红色虚线
# 更多属性可以参考官网：http://matplotlib.org/api/pyplot_api.html
#plt.plot(x, y, 'k', lw=3)

# scatter可以更容易地生成散点图
plt.scatter(x, y_data)


#print(x)
#print(y_data)

#w = np.ones((4,1))
w = np.zeros((4,1))
#print(y_data.mean(axis=0))
y_out = y_data.reshape(y_data.size, 1)
#x_in = np.insert( x.reshape(x.size, 1) , 0, 1, axis=1)
x_in = []
for item in x:
    row =[1, item, item**2, item**3]
    #print(row)
    x_in.append(row)
#x_in = np.insert(x_in, 0, 1, axis=1)

x_in = np.array(x_in, dtype=float)

#print( y_out[2])
size = x.shape[0]

def error(W,X,y):
    #print(X)
    return np.dot(w, X)[0][0]- y_out[i][0]

err = 0.0
loop = 0
alpha = 0.003
diff = 0.000001
c1 = 0.00001
err0 = 0
err1 = 0
err2 = 0
err3 = 0
wreg = np.ones((4,1))
wregval = 1#1# - alpha * c1/size
#print(x_in.shape)
while loop < 500000:
    i = 0
    err = np.zeros((4,1))
    for row in x_in:
        xx = np.array(row).reshape(4,1)
        #reg = c1* np.dot(w.T, w)[0][0]
        #err = err + (np.dot(w.T, xx)[0][0] - y_out[i][0]) * xx
        #print(np.dot(w.T, xx))
        #err =  err + (np.dot(w.T, xx) - y_out[i][0]) *xx
        singerr =  np.dot(w.T, xx) - y_out[i][0]
        w = w - alpha*( singerr * xx)
        #print(w)
        #w = w - alpha*( err * xx)
        i = i + 1
        #print(err)
    #w =  w * wregval - alpha* (err/n)
    #print(w)
    #print(abs(err0 - w[0][0]), abs(err1 - w[1][0]), abs(err2 - w[2][0]), abs(err3 - w[3][0]))
    #break
    if abs(err0 - w[0][0]) <= diff and abs(err1 - w[1][0]) <= diff and abs(err2 - w[2][0]) <= diff and abs(err3 - w[3][0]) <= diff:
        break
    #print(err)
    err0 = w[0][0]
    err1 = w[1][0]
    err2 = w[2][0]
    err3 = w[3][0]
    loop = loop + 1

print(loop, err0,err1, err2, err3, w)

result_y = []
for item in x:
    result_y.append( w[0][0] + item * w[1][0] + w[2][0] * item**2 +  w[3][0] * item**3 )

result_y = np.array(result_y)
plt.scatter(x, result_y)
