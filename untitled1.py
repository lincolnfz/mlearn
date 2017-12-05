# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:59:14 2017

@author: linco
"""
import numpy as np
from sklearn.decomposition import PCA

#零均值化  
def zeroMean(dataMat):        
    meanVal=np.mean(dataMat, axis=0)     #按列求均值，即求各个特征的均值  
    newData=dataMat-meanVal  
    return newData,meanVal  
  
def mypca(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization
  norm_X = X - mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  #print(eig_val)
  
  #print(eig_vec)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  #print(eig_pairs)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  print(feature)
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  return data

mat_a = []
line = 4
while line > 0:
    col = 3
    line_array = []
    while col > 0:
        ran = np.random.randn()+line
        line_array.append(ran);
        col = col - 1
    mat_a.append(line_array);
    line = line -1
    
#print(matt)
    
    
mat = np.array(mat_a)

print(mat)
print(mat[1:,:])

#low = mypca(mat, 1)
#print(low)


pca=PCA(n_components=0.95, svd_solver='full')
#pca.components_ = 0.95
pca.fit(mat)
loww = pca.transform(mat)
#print(pca.explained_variance_)
