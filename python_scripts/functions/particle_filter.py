# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:18:00 2019

@author: matheus.ladvig
"""
import numpy as np


def create_virtual_obs(particle,lambda_values,beta_1):
    
    nb_noise = particle.shape[0]
    noise = np.random.normal(0,1,nb_noise)
    
    observation = particle + beta_1*(np.sqrt(lambda_values)*noise)[...,np.newaxis]
    
    return observation


def calculate_normalized_weigths(particles,obs,lambda_values,beta_1):
    
    m_inv_covar = calculate_inv_noise_covariance_matrix(lambda_values,beta_1)
    weigths = np.zeros(particles.shape[1])
    values = np.zeros(particles.shape[1])
    for i in range(particles.shape[1]):
        
        value = -0.5*(-2*obs.T @ m_inv_covar @ (particles[:,i][...,np.newaxis]) + (particles[:,i][...,np.newaxis].T) @ m_inv_covar @ particles[:,i][...,np.newaxis])
        values[i] = value
        
    
    explosures = np.isinf(values)
    if np.any(explosures):
        print('EXPLODIU')
        weigths[explosures] = 10
        weigths[np.logical_not(explosures)] = 1
    
    else:  
        weigths = np.exp(70)*np.exp(values-np.max(values))    
    
    
    
#    print(weigths)
    weigths = weigths/np.sum(weigths)
#    print('\n')
    
    
    
    return weigths


def resample(weigths):
    
    nb_weigths = weigths.shape[0]
    if np.sum(weigths) < 1:
        weigths[-1] = weigths[-1] + (1 - np.sum(weigths))
    
    indexes = np.random.choice(nb_weigths,nb_weigths, p=weigths)
    
    return indexes

def calculate_inv_noise_covariance_matrix(lambda_values,beta_1):
    
    cov_matrix = np.power(beta_1,2)*np.diag(lambda_values)
    
    
    return np.linalg.inv(cov_matrix)
    

def particle_filter(particles,observation,lambda_values,beta_1):
    
    obs = create_virtual_obs(observation,lambda_values,beta_1)
    weigths = calculate_normalized_weigths(particles,obs,lambda_values,beta_1)
    
    indexes = resample(weigths)
    
    
    return particles[:,indexes]
    
    
    
    
    
    
    
    









    
    
    
    
    