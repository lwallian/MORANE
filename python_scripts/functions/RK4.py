# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:23:36 2019

@author: matheus.ladvig
"""
from deriv_bt import deriv_bt
import numpy as np



def squeeze_python(data):
    
    if len(data.shape)==2:
        return data
    else:
        data = np.squeeze(data)
        return data




def RK4(bt,I,L,C,dt):
    
#   Time integration by 4-th order Runge Kutta 
#   Compute b(t+1)
#   The sizes of the inputs should be :
#   - I = [ I(t) I(t+1) I(t+2)] : 3 x m 
#   - L = [ L(t) L(t+1) L(t+2)] : 3 x m x m
#   - C : m x m x m
#   - bt = b(t) : 1 x m 
#   The result has the size : N x m
    
    
    k1 = deriv_bt(I[0,:][...,np.newaxis],squeeze_python(L[0,:,:]), C, bt)
    k2 = deriv_bt(I[1,:][...,np.newaxis],squeeze_python(L[1,:,:]), C, bt + k1*dt/2)
    k3 = deriv_bt(I[1,:][...,np.newaxis],squeeze_python(L[1,:,:]), C, bt + k2*dt/2)
    k4 = deriv_bt(I[2,:][...,np.newaxis],squeeze_python(L[2,:,:]), C, bt + k3*dt)
    
    b_tp1 = bt + (dt/3)*(k1/2 + k2 + k3 + k4/2)
    
    
    
    return b_tp1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    