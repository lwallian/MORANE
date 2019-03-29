# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:50:08 2019

@author: matheus.ladvig
"""
import numpy as np

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
        pass
    else:
        I = I.T
    
    
    if type(L) is numpy.ndarray:
        if len(L.shape)==2 or len(L.shape)==1:
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    