# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:18:00 2019

@author: matheus.ladvig
"""
import numpy as np


def create_virtual_obs(particle,lambda_values):
    
    nb_noise = particle.shape[0]
    noise = np.random.normal(0,1,nb_noise)
    
    observation = particle + 0.1*(np.sqrt(lambda_values)*noise)[...,np.newaxis]
    
    return observation


def calculate_normalized_weigths(particles,obs,lambda_values):
    
    m_inv_covar = calculate_inv_noise_covariance_matrix(lambda_values)
    weigths = np.zeros(particles.shape[1])
    for i in range(particles.shape[1]):
        
        value = -0.5*(-2*obs.T @ m_inv_covar @ (particles[:,i][...,np.newaxis]) + (particles[:,i][...,np.newaxis].T) @ m_inv_covar @ particles[:,i][...,np.newaxis])
        weigths[i] = np.exp(value[0,0])
    
    weigths = weigths/np.sum(weigths)
    
    
    return weigths


def resample(weigths):
    
    nb_weigths = weigths.shape[0]
    if np.sum(weigths) < 1:
        weigths[-1] = weigths[-1] + (1 - np.sum(weigths))
    
    indexes = np.random.choice(nb_weigths,nb_weigths, p=weigths)
    
    return indexes

def calculate_inv_noise_covariance_matrix(lambda_values):
    
    cov_matrix = 0.1*np.diag(np.sqrt(lambda_values))
    
    
    return np.linalg.inv(cov_matrix)
    

def particle_filter(particles,observation,lambda_values):
    
    obs = create_virtual_obs(observation,lambda_values)
    
    weigths = calculate_normalized_weigths(particles,obs,lambda_values)
    
    indexes = resample(weigths)
    
    
    return particles[:,indexes]
    
    
    
    
    
    
    
    
    
    
    
    
    