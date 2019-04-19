# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:50:08 2019

@author: matheus.ladvig
"""
import numpy as np
from RK4 import RK4

def ismatrix(matrix):
    if type(matrix) is np.ndarray:
        if len(matrix.shape)==2 or len(matrix.shape)==1:
            return True
        else:
            return False
    else:
        if type(matrix) is int or type(matrix) is float:
            return True
        else:
            return False
    

def evol_forward_bt_RK4(I,L,C, dt, bt):
    
#     Compute the next bt
#     The sizes of the inputs should be :
#     - I : 2N-1 x m
#     - L : 2N-1 x m x m
#     - C : m x m x m
#     - bt : N x m 
#     The result has the size : 1 x m
    
    # Modify sizes
    
    N = bt.shape[0]
    
    if I.shape[1] == 1:
        I = np.matlib.repmat(np.transpose(I, (1, 0)),np.max([2*N-1,3]),1)
    else:
        I = I.T
    
    
    if ismatrix(L):
        L = np.tile(np.transpose(L[...,np.newaxis], (2,0,1))[0,:,:],(np.max([2*N-1,3]),1,1))
        
    else:
        L = np.transpose(L[...,np.newaxis],(2,0,1))
    
    
    
    # Time integration by 4-th order Runge Kutta 
    
    if N < 1:
        print('Error: bt is empty')
        return 0
    else:
        bt_evol = RK4(bt[-1,:][np.newaxis,...],I[-3:,:],L[-3:,:,:],C,dt)
    
            

    
            
    return bt_evol
    
    
    
      