# coding=utf-8

'''
作者:Xiaole Wen
程序:多项式曲线拟合算法
'''
import matplotlib.pyplot as plt
import math
import numpy
import random

fig = plt.figure()
ax = fig.add_subplot(111)

#在这里给出拟合多项式的阶数
order=9

#生成曲线上的各个点
x = numpy.arange(-1,1,0.02)
y = [((a*a-1)*(a*a-1)*(a*a-1)+0.5)*numpy.sin(a*2) for a in x]
#ax.plot(x,y,color='r',linestyle='-',marker='')
#,label="(a*a-1)*(a*a-1)*(a*a-1)+0.5"
plt.plot(x,y)
#生成的曲线上的各个点偏移一下，并放入到xa,ya中去
i=0
xa=[]
ya=[]
for xx in x:
    yy=y[i]
    d=float(random.randint(60,140))/100
    #ax.plot([xx*d],[yy*d],color='m',linestyle='',marker='.')
    i+=1
    xa.append(xx*d)
    ya.append(yy*d)

ax.plot(xa,ya,color='m',linestyle='',marker='.')
#存储从0次到m次的所有冥方和
bigMat=[]
for j in range(0,2*order+1):
    sum=0
    for i in range(0,len(xa)):
        sum+=(xa[i]**j)
    bigMat.append(sum)

#计算线性方程组系数矩阵
matA=[]
for rowNum in range(0,order+1):
    row=bigMat[rowNum:rowNum+order+1]
    matA.append(row)

matA=numpy.array(matA)

matB=[]
for i in range(0,order+1):
    ty=0.0
    for k in range(0,len(xa)):
        ty+=ya[k]*(xa[k]**i)
    matB.append(ty)

matB=numpy.array(matB)

matAA=numpy.linalg.solve(matA,matB)

#画出拟合后的曲线
#print(matAA)
xxa= numpy.arange(-1,1.06,0.01)
yya=[]
for i in range(0,len(xxa)):
    yy=0.0
    for j in range(0,order+1):
        dy=(xxa[i]**j)
        dy*=matAA[j]
        yy+=dy
    yya.append(yy)
ax.plot(xxa,yya,color='r',linestyle='-',marker='')

ax.legend()
plt.show()