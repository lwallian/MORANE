# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:21:47 2019

@author: matheus.ladvig
"""

import numpy as np
import matplotlib.pyplot as plt
import PF_class



def apply_dynamic_model(particles):
    
    particles = particles + 0.5 + np.random.randn(particles.shape[0])/2

    return particles

def apply_observation_model(particles):

    hypothesis = particles + np.random.randn(particles.shape[0])/4
    
    return hypothesis
    
def resample(weigths):
    nb_weigths = weigths.shape[0]
    indexes = np.random.choice(nb_weigths,nb_weigths, p=weigths)
    return indexes



def calculate_weigths(state_hypothesis,state_observation):
    
    weigths = np.exp(-0.5*np.power((state_observation*np.ones((state_hypothesis.shape[0])) - state_hypothesis),2))
    
    weigths =  weigths/np.sum(weigths)
    
    
    return weigths
#    
#def first_state_filter(N):
#    
#    particles = np.zeros(N) 
#    
#    return particles
    
    
if __name__ == '__main__':
    
    N = 1000
    
    
    pf = particle_filter(nb_particles = N,\
                        apply_dynamic_model = apply_dynamic_model,\
                        apply_observation_model = apply_observation_model,\
                        resampling = resample,\
                        calculate_weigths = calculate_weigths)
    
    
    
    x = np.arange(1,10,0.5)[...,np.newaxis]
    y = x + np.random.randn(x.shape[0])[...,np.newaxis]
    
    
    state = np.zeros(x.shape[0])
    i = 0
    for obs in y:
        state[i] = pf.update(obs)
        i+=1
        
        
    

    plt.figure()
   
    plt.plot(x[:,0])
    plt.plot(state)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    