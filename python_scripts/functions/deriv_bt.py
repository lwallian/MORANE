# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:29:01 2019

@author: matheus.ladvig
"""
import numpy as np

def deriv_bt(I,L,C, bt):
    
#   Compute the time derivation of bt
#   The sizes of the inputs should be :
#   - I : m
#   - L : m x m
#   - C : m x m x m
#   - bt : N x m 
#   - ord3 (if exist) : m x m x m x m
#   - ord4 (if exist) : m x m x m x mx m
#   The result has the size : N x m
    
    N = bt.shape[0]
    
    
#    bt = np.transpose(bt[:,np.newaxis,np.newaxis,:],(1,2,3,0)) #  m x 1 x 1 x N
    
    bt = bt.T
    
    C = np.multiply(bt,np.transpose(C,(2,0,1)))
    C = np.transpose(C,(1,2,0))
    C = np.sum(C,axis=0)
    
#    C = squeeze(sum(C,1)); % m x m x N
#
#    bt = permute(bt,[1 2 4 3]); % m x 1 x N    
    
    C = np.multiply(bt,C)
    C = np.sum(C,axis=0)[...,np.newaxis]

    
    L = np.multiply(bt,L)
    L = np.sum(L,axis = 0)[...,np.newaxis]
    
    
    
    
    db = -(np.tile(I,(1,N)) + L + C).T
    

    
    
    
    return db