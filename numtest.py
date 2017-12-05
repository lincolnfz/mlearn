#-*- encoding:utf-8 -*-
"""
岭回归example1
@Dylan
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
 
#x是10*10 的Hilbert 矩阵
x=1./(np.arange(1,11)+np.arange(0,10)[:,np.newaxis])
# print(x)
y=np.ones(10)
 
#####compute path
n_alphas=200
alphas=np.logspace(-10,-2,n_alphas)
# print(alphas)
clf=linear_model.Ridge(fit_intercept=False)
coefs=[]
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(x,y)
    coefs.append(clf.coef_)
 
###展示结果
ax=plt.gca()
ax.set_color_cycle(['b','r','g','c','k','y','m'])
 
ax.plot(alphas,coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('ridge coefficients as a function of reqularization')
plt.axis('tight')
plt.show()

def ridgeRegress(xMat,yMat,lam = 0.2):#在没给定lam的时候，默认为0.2
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("这个矩阵是错误的，不能求逆")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws
#对数据进行标准化之后，调用30个不同的lam进行计算
def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat,0)
    xVar = np.var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat,yMat, np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat