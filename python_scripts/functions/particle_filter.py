# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:18:00 2019

@author: matheus.ladvig
"""
import numpy as np
import collections
from evol_forward_bt_MCMC import evol_forward_bt_MCMC
from scipy import optimize
def create_virtual_obs(particle,lambda_values,beta_1):
    
    nb_noise = particle.shape[0]
    noise = np.random.normal(0,1,nb_noise)
    
    observation = particle + beta_1*(np.sqrt(lambda_values)*noise)[...,np.newaxis]
    
    return observation


def calculate_likelihood(particles,obs,lambda_values,beta_1,phi):
    
    m_inv_covar = calculate_inv_noise_covariance_matrix(lambda_values,beta_1)
    weigths = np.zeros(particles.shape[1])
    values = np.zeros(particles.shape[1])
    for i in range(particles.shape[1]):
        
        value = -0.5*phi*(-2*obs.T @ m_inv_covar @ (particles[:,i][...,np.newaxis]) + (particles[:,i][...,np.newaxis].T) @ m_inv_covar @ particles[:,i][...,np.newaxis])
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
    
    
def f(x,*params):
    likelihood,N_threshold,phi = params
    
    likeli_i = np.power(likelihood,(x-1+phi))
    weigths = likeli_i/np.sum(likeli_i)
    ess= 1/np.sum(np.power(weigths,2))
    
    
    return np.abs(ess-N_threshold)
#    return (np.sum(np.power(likelihood,2*(x-1+phi)))/np.power(np.sum(np.power(likelihood,(x-1+phi))),2) - 1/N_threshold)
    
def find_tempering_coeff(likelihood,N_threshold,phi):
    low_inter = 1 - phi
    high_inter = 1
#    TOL = 1e-16
#    MAX_ITER = 1000
#    
#    n = 1
#    
#    while n <= MAX_ITER:
#        
#        c = (low_inter + high_inter)/2
#        value_func = function_optimization(c,likelihood,N_threshold,phi) 
##        print(c)
##        print(value_func)
#        if (value_func== 0) or ((high_inter - low_inter)/2 < TOL):
#            return c
#        
#        n = n + 1
#        
#        if np.sign(low_inter) == np.sign(high_inter):
#            low_inter = c
#        else:
#            high_inter = c
#    
    params = (likelihood,N_threshold,phi)
    rranges = (slice(low_inter,high_inter),)
    resbrute = optimize.brute(f, ranges=rranges, args=params, full_output=True,  finish=optimize.fmin)
    
#    print(resbrute[0])
    return resbrute[0][0]





def calculate_acceptance_prob(particle_candidate,particle,obs,lambda_values,beta_1):
    n_modes = particle_candidate.shape[0]
    vector = 1*np.ones(int(n_modes-2))
    vector = np.concatenate((np.array([1,1]),vector))
    
    
    m_inv_covar = calculate_inv_noise_covariance_matrix(lambda_values,beta_1) @ np.diag(vector)
    #### likeli candidate
    value_likeli_candi = np.exp(-0.5*(-2*obs.T @ m_inv_covar @ particle_candidate + particle_candidate.T @ m_inv_covar @ particle_candidate))
    #### likeli current
    value_likeli_curre = np.exp(-0.5*(-2*obs.T @ m_inv_covar @ particle + particle.T @ m_inv_covar @ particle))

    if (value_likeli_curre==0):
        ratio = 0
    else:
        ratio = value_likeli_candi/value_likeli_curre
    
        
    prob = np.min([1,ratio])
    
    
    return prob
    
    
    

#def mutation_MCMC(particles,var_coef,obs,lambda_values,beta_1,mean_coef):
#    M=20
#    pho = 0.2
#    for index_part in range(particles.shape[1]):
#        particle = particles[:,index_part][...,np.newaxis]
#        for j_mutation in range(M):
#            noise = np.diag(var_coef) @ np.random.normal(size=(var_coef.shape[0],1))
#            particle_candidate = mean_coef[...,np.newaxis] + pho*(particle -  mean_coef[...,np.newaxis]) +np.sqrt(1-pho**2)*noise
#            prob_acept = calculate_acceptance_prob(particle_candidate,particle,obs,lambda_values,beta_1)
##            if prob_acept>0.01:
##                print(prob_acept)
#            
#            
#            
#            if np.random.uniform(0,1)<prob_acept:
#                particle = particle_candidate
##                print('mutou com prob: '+ str(prob_acept))
#                
#        particles[:,index_part] = particle[:,0]
#    
#    
#    return particles
    
#    
    
def propagate_particle(pho,particle_past_original,noise_accepted_original,delta_t,I,L,C, pchol_cov_noises, dt):
    
    
    bt_candidate = particle_past_original[...,np.newaxis]
    noise_candidate_complete = np.zeros(shape = noise_accepted_original.shape)
    
    for iter_i in range(delta_t):
        bt_candidate,noise_candidate = evol_forward_bt_MCMC(I,L,C, pchol_cov_noises, dt,bt_candidate,0,0,True,noise_accepted_original[iter_i,:],pho)
        if iter_i==0:
            noise_candidate_complete = noise_candidate
        else: 
            noise_candidate_complete = np.concatenate((noise_candidate_complete,noise_candidate),axis=0)
    
    
    
    return bt_candidate,noise_candidate_complete


   
    
def mutation_MCMC_from_past(pho,M,particle,noise,particle_past,delta_t,I,L,C, pchol_cov_noises, dt,obs,lambda_values,beta_1):
#    M = 10
    particle_accepted =  particle
    noise_accepted = noise
    for iter_i in range(M):
        
        particle_candidate,noise_candidate = propagate_particle(pho,particle_past,noise_accepted,delta_t,I,L,C, pchol_cov_noises, dt)
        prob_acept = calculate_acceptance_prob(particle_candidate[:,0],particle_accepted,obs,lambda_values,beta_1)
#        print(prob_acept)
        if np.random.uniform(0,1)<prob_acept:
#            print(prob_acept)
            particle_accepted = particle_candidate[:,0]
            noise_accepted = noise_candidate
    

        
        
    return particle_accepted,noise_accepted

    
    
def find_tempering_coeff_brute(likelihood,N_threshold,phi):
    
    phis = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.5,0.4,0.3,0.2,0.1,0.001]
    
    out = False
    i=0
    while out==False:
        
        likeli_i = np.power(likelihood,(phis[i]-1+phi))
        weigths = likeli_i/np.sum(likeli_i)
        ess= 1/np.sum(np.power(weigths,2))
        if (ess>N_threshold) or (i==len(phis)):
            out=True

        else:
            i+=1
        
        
    return phis[i]
    
    
    
def particle_filter(M,ILC_a_cst,pchol_cov_noises,dt,delta_t,particles,observation,lambda_values,beta_1,N_threshold,noises,particles_past,pho):
    I = ILC_a_cst['modal_dt']['I']
    L = ILC_a_cst['modal_dt']['L']
    C = ILC_a_cst['modal_dt']['C']

    only_repeated = True
    obs = create_virtual_obs(observation,lambda_values,beta_1)
    likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1,1)
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
    phi = 1
    phi_guime = 0
    ess = calculate_effective_sample_size(weigths)
    while (r<3):
        
        
        if (ess>N_threshold):
            index_print = np.where((weigths>0.05))
            print('indexes with more than 5% probability of sampling: '+ str(index_print[0]))
            print('phi guime: '+ str(phi_guime))
            
            phi = 1
#            print(len(collections.Counter(indexes).keys()))  
            print('Number of temp iter: '+str(r))
            print('\n')
            return particles,obs
        else:
            
            phi_guime = find_tempering_coeff(likelihood,N_threshold,phi)
            if phi_guime>0.99:
                phi_guime = 0.99
                
            index_print = np.where((weigths>0.05))
            
            print('indexes with more than 5% probability of sampling: '+ str(index_print[0]))
            print('phi guime: '+ str(phi_guime))
            
#        weigths = np.power(weigths,(phi - phi_before))/np.sum(np.power(weigths,(phi - phi_before)))
        print(phi_guime-1+phi)
        likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1,phi_guime-1+phi)
        weigths = likelihood/np.sum(likelihood)
        ess = calculate_effective_sample_size(weigths)
        print('ESS: '+ str(ess))
#        dict_in = {}
#        print(phi)
#        for index in np.where((weigths>=0.1*np.max(weigths))):
#            dict_in[str(index)] = weigths[index]        
#        print('Dict with the most probable indexes: '+ str(dict_in))
        indexes = resample(weigths)
        
        ################################################################################
        ###################################----MUTATION----#############################
        if only_repeated ==True:
            counter = collections.Counter(indexes)
            keys = counter.keys()
            indexes_repeated = []
            for key in keys:
                if counter[key]>0:
                    indexes_repeated.append(key)
        
        particles_selected = np.zeros(shape=particles.shape)
        noises_selected = np.zeros(shape=noises.shape)
        i = 0
        for index in indexes:
            if index in indexes_repeated:
                noise = noises[...,index]
                particle = particles[:,index]
                particle_past = particles_past[:,index]
                particles_selected[...,i],noises_selected[...,i] = mutation_MCMC_from_past(pho,M,particle,noise,particle_past,delta_t,I,L,C, pchol_cov_noises,\
                                                                                          dt,obs,lambda_values,beta_1)
            else:
                particles_selected[...,i] = particles[...,index]
                noises_selected[...,i] = noises[...,index]
                
            i+=1
            
        
        ############################################################################################
        #################################---UPDATE---###############################################
        particles = particles_selected
        noises = noises_selected
        
        
        phi = 1 - phi_guime
        r += 1
        
#        particles = particles[:,indexes]
#        noises = noises[...,indexes]
#        particles_past = particles_past[:,indexes]
#        
##        particles = mutation_MCMC(particles[:,indexes],var_coef,obs,lambda_values,beta_1,mean_coef)
#        particles,noises = mutation_MCMC_from_past(particles,indexes,noises,only_repeated,particles_past)
#        likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1)
        
#        particles = particles[:,indexes]
#        likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1)
        
     
        
        
        
#    print(collections.Counter(indexes))    
   
    
    print('\n')
    
    
    return particles,obs
    
    


    
    
    
    
    
    
    
    









    
    
    
    
    