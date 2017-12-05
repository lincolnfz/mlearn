# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:03:40 2017

@author: linco
"""
import math
import numpy as np

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
print(mat.T)


p=[0.5, 0.25, 0.125, 0.125]
sum = 0;
for i in p:
    sum = sum + ( 0 - (i * np.log2(i)) )
    
print(sum)