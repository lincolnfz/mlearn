# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:40:58 2018

@author: linco
"""

import numpy as np

#
pi = np.array([0.2, 0.4, 0.4])

#
b = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])

#转移概率
a = np.array([[0.5, 0.2, 0.3],[0.3, 0.5, 0.2],[0.2, 0.3, 0.5]])

#观测序列
seq = np.array([0, 1, 0])

class hmm(object):

    #前向算法
    def forward(self, start_prob, move_prob, oserver_prob, seq):
        alpha = np.zeros(shape = (pi.shape[-1], seq.shape[-1]))
        idx = 0
        for i in seq:
            if idx == 0:
                val = np.multiply(start_prob.T, oserver_prob.T[i])
                alpha[0:,idx] = val
            else :
                #print(idx)
                a_vec = alpha.T[idx-1]
                a_vec = a_vec.reshape(1, -1)
                a_sum = np.dot(a_vec, move_prob)
                alpha[0:,idx] = np.multiply(a_sum, oserver_prob.T[i])
            idx += 1
        prob = np.sum(alpha[:,-1])
        return alpha, prob
    
    #后向算法
    def backward(self, start_prob, move_prob, oserver_prob, seq):
        seq = seq[-1::-1]
        beta = np.ones(shape= (move_prob.shape[-1], seq.shape[-1]))
        beta_tmp = np.ones(shape= (move_prob.shape[-1], 1))
        idx = 0
        for i in seq:
            beta_tmp = np.multiply(beta_tmp.T, oserver_prob.T[i]).T
            beta_tmp = np.dot(move_prob, beta_tmp)
            beta[0:, beta.shape[1]-idx-2] = beta_tmp.ravel()
            idx += 1
            if idx >= beta.shape[1]-1:
                break
        prob =  np.multiply(beta.T[0], oserver_prob.T[seq[-1]].T )
        prob = np.sum(np.multiply(prob, start_prob))
        return beta, prob
    
    def r(self, alpha, beta):
        ri = np.multiply(alpha, beta)
        return np.true_divide(ri, np.sum(ri, axis=0))
    
    def prob_seq(self, alpha, move_prob, oserver_prob, beta, seq):
        #den = np.sum(alpha[:,-1])
        idx_i = 0
        n_prob = list()
        for i in seq:
            b_temp = np.multiply(oserver_prob.T[seq[idx_i+1]], beta.T[seq[idx_i+1]])
            #print(b_temp.shape)
            alpha_tmp = alpha[::,idx_i]
            idx_j = 0
            state_prob = list()
            for j in alpha_tmp:
                #print(j)
                #print(np.multiply(move_prob[idx_j], b_temp))
                #print(np.multiply(move_prob[idx_j], b_temp) * j)
                state_prob.append(np.multiply(move_prob[idx_j], b_temp) * j)                
                idx_j += 1
            state_prob = np.array(state_prob)
            state_prob = state_prob / np.sum(state_prob)
            n_prob.append(state_prob)
            #print(move_prob[0])
            #alpha_tmp = alpha_tmp.reshape(alpha_tmp.shape[-1], -1)
            #print(alpha_tmp.shape)
            
            idx_i += 1
            #print(idx_i, seq.shape[-1])
            if idx_i >= seq.shape[-1]-1:
                #idx_i += 1
                break;
        n_prob = np.array(n_prob)
        #print(n_prob)
        return n_prob
    
    def viterbi(self, start_prob, move_prob, oserver_prob, seq):
        zeta = list()
        psi = list()
        zeta_tmp = start_prob * oserver_prob.T[seq[0]]
        zeta.append( zeta_tmp )
        psi_tmp = np.zeros(shape=(start_prob.shape[-1]))
        psi.append(psi_tmp)
        idx_i = 0
        for i in seq:
            if idx_i == 0:
                idx_i += 1
                continue
            idx_j = 0
            zeta_tmp_l = list()
            #psi_tmp_pos = list()
            psi_tmp = []
            for j in range(start_prob.shape[-1]):
                psi_tmp_l = zeta_tmp * move_prob.T[idx_j]
                zeta_max = np.max(psi_tmp_l)
                re = np.where(psi_tmp_l == np.max(psi_tmp_l))
                #print(re[0][0])
                psi_tmp.append(re[0][0])
                zeta_tmp_l.append(zeta_max * oserver_prob[idx_j][i] )
                idx_j += 1
            zeta_tmp = np.array(zeta_tmp_l)
            zeta.append(zeta_tmp)
            psi.append(np.array(psi_tmp))
            idx_i += 1
        
        ichain = list()
        last_i  = np.where(zeta[-1] == np.max(zeta[-1]))
        last_i = last_i[0][0]
        ichain.append(last_i)
        
        
        for i in psi[-1::-1]:
            last_i = i[last_i]
            ichain.append(last_i)
        
        ichain.pop()
        ichain = ichain[-1::-1]
        #print(ichain)
        #print(zeta[-1])
        
        zeta = np.array(zeta).T
        psi = np.array(psi).T
        
        #print(np.max(zeta, axis=0)[-1])
        #print(zeta)
        #print(psi)
        return zeta, psi, ichain

if __name__ == '__main__':
    test = hmm()
    alpha, prob = test.forward(pi, a, b, seq)
    #print(alpha)
    print(prob)
    beta, prob = test.backward(pi, a, b, seq)
    #print(prob)
    r = test.r(alpha, beta)
    #print(r)
    n_prob = test.prob_seq(alpha, a, b, beta, seq)
    print(n_prob)
    #print( np.sum(n_prob, axis = 1) )
    #oo = np.sum(n_prob, axis = 1)
    #print( np.sum(oo, axis = 1) )
    test.viterbi(pi, a, b, seq)
    #print(np.sum(n_prob, axis=2))
    #print(np.sum(n_prob, axis=2))
    #dd = np.sum(n_prob, axis=2)
    #print(np.sum(dd, axis=1))
    #df = np.sum(dd, axis=1)
    