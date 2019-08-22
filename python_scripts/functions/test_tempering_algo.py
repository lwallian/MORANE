# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:57:11 2019

@author: matheus.ladvig
"""

import numpy as np
from scipy import optimize


def f(x,*params):
    likelihood,N_threshold = params
    
    likeli_i = np.power(likelihood,(x))
    weigths = likeli_i/np.sum(likeli_i)
    ess= 1/np.sum(np.power(weigths,2))
    
    
    return np.abs((N_threshold-ess))
    
    
    
    
#    return (np.sum(np.power(likelihood,2*(x-1+phi)))/np.power(np.sum(np.power(likelihood,(x-1+phi))),2) - 1/N_threshold)
    
def find_tempering_coeff(likelihood,N_threshold):
    low_inter = 0.1
    high_inter = 0.5

    params = (likelihood,N_threshold)
    rranges = (slice(low_inter,high_inter,0.0001),)
    resbrute = optimize.brute(f, ranges=rranges, args=params, full_output=True,  finish=optimize.fmin)
    
    print(resbrute[0])
    return resbrute




likeli = np.array([0.1,0.1,0.1,0.1,0.1,0.000002,0.000215,0.0001544,0.0002454,0.000155487])

weigths = likeli/np.sum(likeli)

ess = 1/np.sum(np.power(weigths,2))
print(ess)

phis=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]#,0.1,0.01,0.001,0.0001]
for i in phis:
    likeli_i = np.power(likeli,i)
    weigths = likeli_i/np.sum(likeli_i)
    print(1/np.sum(np.power(weigths,2)))


N_threshold = 6
phi = 1
phi_guime = find_tempering_coeff(likeli,N_threshold)

likeli_i = np.power(likeli,(phi_guime[0]-1+phi))
weigths = likeli_i/np.sum(likeli_i)
ess= 1/np.sum(np.power(weigths,2))

print(ess)
