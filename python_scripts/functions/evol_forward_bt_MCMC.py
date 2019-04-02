# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:36:02 2019

@author: matheus.ladvig
"""
import numpy as np
def evol_forward_bt_MCMC(I,L,C, pchol_cov_noises, dt, bt,bt_fv,bt_m):
    
#    Compute the next bt
#    The sizes of the inputs should be :
#    - I : m
#    - L : m x m
#    - C : m x m x m
#    - bt : 1 x m x nb_pcl
#    The result has the size : 1 x m
    
    
    n,nb_pc1 = bt.shape
    
    bt = bt[:,np.newaxis,np.newaxis,:]
    
    C = np.multiply(np.transpose(C,(2,0,1)),np.transpose(bt,(3,2,0,1)))
    C = np.transpose(C,(2,3,1,0))
    C = np.sum(C,axis=0)
    
    C = np.multiply(np.transpose(C,(2,0,1))[:,np.newaxis,:,:],np.transpose(bt,(3,2,0,1)))
    C = np.transpose(C[:,0,:,:],(1,2,0))
    C = np.sum(C,axis=0)
    C = C[:,np.newaxis,np.newaxis,:]
    
    L = np.multiply(L,np.transpose(bt,(3,2,0,1)))
    L = np.transpose(L,(2,3,1,0))
    L = np.sum(L,axis=0)
    
    L = L[:,np.newaxis,:,:]
    
    
    d_b_fv = - np.add(I,np.transpose(L+C,(3,2,0,1)))*dt
    d_b_fv = np.transpose(d_b_fv,(2,3,1,0))
    
    del I
    del L
    del C
    
    
    noises = np.matmul(pchol_cov_noises,np.random.normal(size=((n+1)*n,nb_pc1)))*np.sqrt(dt)
    noises = noises[:,np.newaxis,np.newaxis,:]
    
    del pchol_cov_noises
    
    theta_alpha0_dB_t = noises[:n,:,:,:]
    alpha_dB_t = np.reshape(noises[n:,:,:,:],(n,n,1,nb_pc1))
    
    del noises
    
    #############
    
    
    #############
    
    
    alpha_dB_t = np.multiply(np.transpose(bt,(3,2,0,1)),np.transpose(alpha_dB_t,(3,2,0,1)))
    alpha_dB_t = np.transpose(alpha_dB_t,(2,3,1,0))
    alpha_dB_t = np.sum(alpha_dB_t,0)
    alpha_dB_t = alpha_dB_t[:,:,np.newaxis,:]
    
    
    d_b_m = alpha_dB_t + theta_alpha0_dB_t
    
    #### if nargin>6
    bt_fv = np.add(bt_fv, d_b_fv[:,0,0,:])
    bt_fv = bt_fv[np.newaxis,...]

    bt_m  = np.add(bt_m, d_b_m[:,0,0,:] )
    bt_m = bt_m[np.newaxis,...]
    
    bt_evol = np.add(bt[:,0,0,:],d_b_fv[:,0,0,:],d_b_m[:,0,0,:])
    bt_evol = bt_evol[np.newaxis,...]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return bt_evol,bt_fv,bt_m
    
    
    
    
    
    
    
    
    
    