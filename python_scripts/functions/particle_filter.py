# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:18:00 2019

@author: matheus.ladvig
"""
import numpy as np
import collections

def create_virtual_obs(particle,lambda_values,beta_1):
    
    nb_noise = particle.shape[0]
    noise = np.random.normal(0,1,nb_noise)
    
    observation = particle + beta_1*(np.sqrt(lambda_values)*noise)[...,np.newaxis]
    
    return observation


def calculate_likelihood(particles,obs,lambda_values,beta_1):
    
    m_inv_covar = calculate_inv_noise_covariance_matrix(lambda_values,beta_1)
    weigths = np.zeros(particles.shape[1])
    values = np.zeros(particles.shape[1])
    for i in range(particles.shape[1]):
        
        value = -0.5*(-2*obs.T @ m_inv_covar @ (particles[:,i][...,np.newaxis]) + (particles[:,i][...,np.newaxis].T) @ m_inv_covar @ particles[:,i][...,np.newaxis])
        values[i] = value
        
    
    explosures = np.isinf(values)
    if np.any(explosures):
        print('EXPLOSED: Recalculating weigths')
        
        weigths[explosures] = 10
        weigths[np.logical_not(explosures)] = 1
    
    else:  
        weigths = np.exp(70)*np.exp(values-np.max(values))    
    
    
    
#    print(weigths)
    
    
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
    
def calculate_effective_sample_size(weigths):
    
    
    return 1/np.sum(np.power(weigths,2))
    
    
def function_optimization(value,likelihood,N_threshold):

    return (np.sum(np.power(likelihood,2*value))/np.power(np.sum(np.power(likelihood,value)),2) - 1/N_threshold)
    
def find_tempering_coeff(likelihood,N_threshold,phi):
    low_inter = phi
    high_inter = 1
    TOL = 1e-16
    MAX_ITER = 1000
    
    n = 1
    
    while n <= MAX_ITER:
        
        c = (low_inter + high_inter)/2
        value_func = function_optimization(c,likelihood,N_threshold) 
#        print(c)
#        print(value_func)
        if (value_func== 0) or ((high_inter - low_inter)/2 < TOL):
            return c
        
        n = n + 1
        
        if np.sign(low_inter) == np.sign(high_inter):
            low_inter = c
        else:
            high_inter = c
    
    
    
    
    return c  



def calculate_weigths(likelihood,phi):
    
    
    
    pass


    
def particle_filter(particles,observation,lambda_values,beta_1,N_threshold):
    
    obs = create_virtual_obs(observation,lambda_values,beta_1)
    likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1)
#    print(weigths)
#    print('Number of effective particles: '+ str(1/np.sum(np.power(weigths,2))))
    
    
#    if 1/np.sum(np.power(weigths,2))>2:
#    dict_in = {}
#    
#    for index in np.where((weigths>=0.1*np.max(weigths))):
#        dict_in[str(index)] = weigths[index]
#        
#    print('Dict with the most probable indexes: '+ str(dict_in))
    
    
    #    index_MAP = np.where((weigths==np.max(weigths)))
    
#    particle_estimate = np.sum(particles*np.tile(weigths[np.newaxis,...],(particles.shape[0],1)),axis=1)
    
    
#    weigths = (likelihood*weigths_time_past)/np.sum(likelihood*weigths_time_past)
    weigths = likelihood/np.sum(likelihood)
#    ESS = calculate_effective_sample_size(weigths)
    r = 0
    phi = 0
    phi_before = 0
    while ((phi<1) or (r<10)):
        r += 1
        ess = calculate_effective_sample_size(weigths)
        if ess>N_threshold:
            print(ess)
            phi_before = phi
            phi = 1
#            print(len(collections.Counter(indexes).keys()))    
            print('\n')
            return particles
        else:
            phi_before = phi
            phi = find_tempering_coeff(likelihood,N_threshold,phi)
            
        weigths = np.power(weigths,(phi - phi_before))/np.sum(np.power(weigths,(phi - phi_before)))
#        dict_in = {}
#        print(phi)
#        for index in np.where((weigths>=0.1*np.max(weigths))):
#            dict_in[str(index)] = weigths[index]        
#        print('Dict with the most probable indexes: '+ str(dict_in))
        indexes = resample(weigths)
        
        
        
        
        particles = particles[:,indexes]
        likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1)
        
     
        
        
        
#    print(collections.Counter(indexes))    
   
    
#    print('\n')
    
    
    return particles
    
    


    
    
    
    
    
    
    
    









    
    
    
    
    