# -*- coding:utf-8 -*-
__author__ = 'lincolnfz@gmail.com'

import numpy as np
import math
from sympy import *


def gen(tims):
    for i in range(tims):
        yield i
        #print(x)
    print('end.')

if __name__ == '__main__':
    mata = np.mat(np.array([[1.0,2.0],[2.0,4.0]]))
    matb = np.mat(np.array([[4.0],[8.0]]))
    #mataa = np.linalg.solve(mata, matb)
    #print (mataa)
    '''if  np.linalg.det(mata) == 0.0:
        print ("no ma")
        return

    matt =  mata.I * matb
    print (matt)'''
    x = np.mat([[4, 2, -5], [6,4,-9],[5,3,-7]])
    eigvalue, eigvector = np.linalg.eig(x)
    #ret = np.mat(eigvector) .T* np.mat(np.array([[6.,0.], [0., -1.] ])) * np.mat(eigvector)
    print (eigvalue, eigvector)
    #print (ret)
    
    f = gen(2)
    x = next(f)
    print(x)
    x = next(f)
    print(x)
    
    x = {'name':'lin', 'add':'fz'}
    name = x
    print(name)
    
    x=Symbol("x")
    #di = diff(1/(1+x**2),x)
    
    #b = 8
    #w = 9
    d2 = diff(exp(1**(-c*((x-c2)**2)), x))
    print(d2)