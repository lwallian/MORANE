# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:51:23 2019

@author: matheus.ladvig
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


#path = 'Users\matheus.ladvig\Desktop\mecanique de fluides\results'
#pth = Path(__file__).parents[1]


modes = [2,4,6,8,16]
plt.figure()

for index,mode in enumerate(modes):
    
    
    path_1 = Path(r'C:/Users\matheus.ladvig\Desktop\mecanique de fluides\results/variance in the modes convergence/variance_des_bs_50_80_500/variances_10_to_100_'+str(mode)+'modes.npy')
    path_2 = Path(r'C:/Users\matheus.ladvig\Desktop\mecanique de fluides\results/variance in the modes convergence/variance_des_bs_dans_inegral_premier_periode_10_100_1000/variances_10_to_100_'+str(mode)+'modes.npy')
    
    
    var_1 = np.load(str(path_1))[0,:]
    var_2 = np.load(str(path_2))[0,:]
    
    var = [var_2[0],var_1[0],var_1[1],var_2[1],var_1[2],var_2[2]]
    part = np.log10(np.array([1,10,50,80,100,500,1000]))
    
    if mode in [2,4,6]:
        plt.subplot(2,3,(index+1))
    else:
        plt.subplot(2,2,(index))
    
    plt.plot(part,np.hstack((0,np.array(var))),'r-.o')
    plt.xlabel('Log10 of particles')
    
    plt.ylabel('Chronos variance in '+ str(mode)+' modes')
    
#    plt.ylabel('Variance in the modes of '+r'$b_{'+str(int(mode))+'}$(t)')
    plt.grid()
