# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:18:00 2019

@author: matheus.ladvig
"""
import numpy as np
import collections
from evol_forward_bt_MCMC import evol_forward_bt_MCMC
from scipy import optimize

#def create_virtual_obs(particle,lambda_values,beta_1):
#    
#    nb_noise = particle.shape[0]
#    noise = np.random.normal(0,1,nb_noise)
#    
#    observation = particle + beta_1*(np.sqrt(lambda_values)*noise)[...,np.newaxis]
#    
#    return observation


#def calculate_likelihood(particles,obs,lambda_values,beta_1,phi):
#    
#    m_inv_covar = calculate_inv_noise_covariance_matrix(lambda_values,beta_1)
#    weigths = np.zeros(particles.shape[1])
#    values = np.zeros(particles.shape[1])
#    for i in range(particles.shape[1]):
#        
#        value = -0.5*phi*(-2*obs.T @ m_inv_covar @ (particles[:,i][...,np.newaxis]) + (particles[:,i][...,np.newaxis].T) @ m_inv_covar @ particles[:,i][...,np.newaxis])
#        values[i] = value
#        
#    
#    explosures = np.isinf(values)
#    if np.any(explosures):
#        print('EXPLOSED: Recalculating weigths')
#        
#        weigths[explosures] = 10
#        weigths[np.logical_not(explosures)] = 1
#    
#    else:  
#        weigths = np.exp(70)*np.exp(values-np.max(values))    
#    
#    
#    
##    print(weigths)
#    
#    
##    print('\n')
#    
#    
#    
#    return weigths


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





#def calculate_acceptance_prob(particle_candidate,particle,obs,lambda_values,beta_1):
#    n_modes = particle_candidate.shape[0]
#    vector = 1*np.ones(int(n_modes-2))
#    vector = np.concatenate((np.array([1,1]),vector))
#    
#    
#    m_inv_covar = calculate_inv_noise_covariance_matrix(lambda_values,beta_1) @ np.diag(vector)
#    #### likeli candidate
#    value_likeli_candi = np.exp(-0.5*(-2*obs.T @ m_inv_covar @ particle_candidate + particle_candidate.T @ m_inv_covar @ particle_candidate))
#    #### likeli current
#    value_likeli_curre = np.exp(-0.5*(-2*obs.T @ m_inv_covar @ particle + particle.T @ m_inv_covar @ particle))
#
#    if (value_likeli_curre==0):
#        ratio = 0
#    else:
#        ratio = value_likeli_candi/value_likeli_curre
#    
#        
#    prob = np.min([1,ratio])
#    
#    
#    return prob
    
    


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


   
    
#def mutation_MCMC_from_past(pho,M,particle,noise,particle_past,delta_t,I,L,C, pchol_cov_noises, dt,obs,lambda_values,beta_1):
##    M = 10
#    particle_accepted =  particle
#    noise_accepted = noise
#    for iter_i in range(M):
#        
#        particle_candidate,noise_candidate = propagate_particle(pho,particle_past,noise_accepted,delta_t,I,L,C, pchol_cov_noises, dt)
#        prob_acept = calculate_acceptance_prob(particle_candidate[:,0],particle_accepted,obs,lambda_values,beta_1)
##        print(prob_acept)
#        if np.random.uniform(0,1)<prob_acept:
##            print(prob_acept)
#            particle_accepted = particle_candidate[:,0]
#            noise_accepted = noise_candidate
#    
#
#
#    return particle_accepted,noise_accepted

    
    
#def find_tempering_coeff_brute(likelihood,N_threshold,phi):
#    
#    phis = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.5,0.4,0.3,0.2,0.1,0.001]
#    
#    out = False
#    i=0
#    while out==False:
#        
#        likeli_i = np.power(likelihood,(phis[i]-1+phi))
#        weigths = likeli_i/np.sum(likeli_i)
#        ess= 1/np.sum(np.power(weigths,2))
#        if (ess>N_threshold) or (i==len(phis)):
#            out=True
#
#        else:
#            i+=1
#        
#        
#    return phis[i]


def create_noise_champ(shape_noise):
    
    return np.random.normal(size=(shape_noise))
    
def create_virtual_champs_vitesse_DNS(champ_orig,H,L,dimensions,std_space,dX,subsampling_grid):
    
#    shape_noise = champ.shape
#    noise = create_noise_champ(shape_noise)
#    h = [0.95,0.05]
#    champ_obs = np.zeros(shape=champ_orig.shape)
    champ_obs_1 = np.zeros(shape=champ_orig.shape)
    x_size = dimensions['x_size']
    y_size = dimensions['y_size']
    z_size = dimensions['z_size']
    dim    = dimensions['dim'] 
    
    
    
    # If the flow is going in the direction of x
    number_of_points_correlated = int(std_space/(dX[0,0]*subsampling_grid))
    dist = dX[0,0]*np.arange(-number_of_points_correlated,number_of_points_correlated+1,1)
    h = np.exp(-(dist**2)/(2*std_space**2))
    sum_h = np.sum(h)
    
    h = h/sum_h
    
    
    
    champ = np.reshape(champ_orig,(x_size,y_size,z_size,dim),order='F')
    
    
    
    
    
    
    
    
    
    
    
    
#    i = 0 
#    j = 0
#    for d in range(dim):
##        print(i)
#        for z in range(z_size):
#            for y in range(y_size):
#                
#                
#                a = champ[1:number_of_points_correlated+1,y,z,d]
#                b = np.flip(a,axis=0)
#                
#                c = champ[-number_of_points_correlated-1:-1,y,z,d]
#                dc = np.flip(c,axis=0)
#                
#                signal = np.concatenate((b,champ[:,y,z,d]),axis=0)
#                signal = np.concatenate((signal, dc),axis=0)
#                
#                champ_obs_1[j*x_size:(j+1)*x_size] = np.convolve(signal,h/sum_h,'valid')
#                j+=1
#                
#                for x in range(x_size):
#                    
#                    if x==0:
#                        champ_obs[i] = h[0]*champ[x,y,z,d] + 2*h[1]*champ[x+1,y,z,d]
#                        
#                    if x == (x_size-1):
#                        champ_obs[i] = 2*h[1]*champ[x-1,y,z,d] + h[0]*champ[x,y,z,d]
#                        
#                    else:
#                        champ_obs[i] = h[1]*champ[x-1,y,z,d] + h[0]*champ[x,y,z,d] + h[1]*champ[x+1,y,z,d]
#                    
#                
#                    i+=1
                    
                    
                    
    shape_noise = champ_obs_1.shape
    noise = np.random.normal(size=(shape_noise))
    champ_obs_1 = champ_obs_1 + L * noise
    
#    if type(H) is int:
#        noise = L*noise
#        obs = H*champ + noise
#    else:
#        obs = np.matmul(H,champ) + np.matmul(L,noise)

    return champ_obs_1

def calculate_inverse_covariance(L):
    
    sigma = np.matmul(L.T,L)
    
    sigma_inverse = np.linalg.inv(sigma) 
    
    
    return sigma_inverse

def calculate_champs_likelihood(particles_chronos,obs_K,phi,Hr_K):
    
    
    
    likeli = np.zeros(shape=particles_chronos.shape[1])
    values = np.zeros(shape=particles_chronos.shape[1])
    
    
    particles_chronos = np.vstack((particles_chronos,np.ones((1,particles_chronos.shape[1]))))
    
    a = obs_K @ particles_chronos
    c = np.einsum('ij,ji->i', (particles_chronos.T @ Hr_K), particles_chronos) 
    values = -2*a + c
    
    values = -0.5*phi*values
    
    
    
    
    
    
    
    
##    for i in range(particles_chronos.shape[1]):
##        sigma_H = np.matmul(sigma_inverse,H)
##        K = np.matmul(sigma_H,np.matmul(topos,particles_chronos[:,i][...,np.newaxis]))
##        
##        b_T_mult_H_T = np.matmul((particles_chronos[:,i][...,np.newaxis]).T,H.T)
##        value_i = -2*np.matmul(obs.T,K) + np.matmul(b_T_mult_H_T,K)
##        
##        values[i] = -0.5*value_i
#    
#    H_phi_b = np.matmul(matrix_H,particles_chronos)
#    d = H_phi_b[:100000,:]
##    K = sigma_inverse*H_phi_b
##    b_T_mult_H_T = H_phi_b.T
##    del H_phi_b
##    a = np.matmul(b_T_mult_H_T,K)
#    a = np.einsum('ij,ji->i', H_phi_b.T, sigma_inverse*H_phi_b)
##    del b_T_mult_H_T
#    b = -2*np.matmul(obs.T,sigma_inverse*H_phi_b)[0,:]
##    del K
#    values_vrai = b + a
##    res_attendu = -sigma_inverse*obs.T@obs 
##    del a
#    del b
#    teste = -0.5*phi*(values_vrai + res_attendu[0,0])
#    values = -0.5*phi*values_vrai
    
    
  
#    sigma_H = sigma_inverse
#    for i in range(particles_chronos.shape[1]):
#        
##        K = sigma_H*np.matmul(topos,particles_chronos[:,i][...,np.newaxis])
#        
#        K = sigma_inverse*np.matmul(matrix_H,particles_chronos[:,i][...,np.newaxis])
#
##        b_T_mult_H_T = np.matmul((particles_chronos[:,i][...,np.newaxis]).T,topos.T)
#        b_T_mult_H_T = (matrix_H @ (particles_chronos[:,i][...,np.newaxis])).T
#        
#        value_i = -2*np.matmul(obs.T,K) + np.matmul(b_T_mult_H_T,K)
#        
#        values[i] = -0.5*phi*value_i
##        print(i)
        
        
        
    explosures = np.isinf(values)
    if np.any(explosures):
        print('EXPLOSED: Recalculating weigths')
        
        likeli[explosures] = 5
        likeli[np.logical_not(explosures)] = 1
    
    else:  
        
        b = values-np.max(values)
        
#        c = teste-np.max(teste)
        
        likeli = np.exp(90)*np.exp(b)   
        
#        likeli_teste = np.exp(90)*np.exp(c)
        
        

    return likeli



   
#def calculate_acceptance_prob_champ(particle_candidate,particle_accepted,obs,H,L,topos,sigma_inverse,matrix_H):
#    
#   
#   
#    
#    
#    particle_candidate = np.vstack((particle_candidate[...,np.newaxis],np.ones((1,particle_candidate[...,np.newaxis].shape[1]))))
#    particle_accepted = np.vstack((particle_accepted[...,np.newaxis],np.ones((1,particle_accepted[...,np.newaxis].shape[1]))))
#    
#    
#    sigma_H = sigma_inverse*H
#    
##    K = sigma_H*np.matmul(topos,particle_candidate)  
##    b_T_mult_H_T = np.matmul((particle_candidate).T,topos.T)
##    value_candidate = -2*np.matmul(obs.T,K) + np.matmul(b_T_mult_H_T,K)
#    K = sigma_H*np.matmul(matrix_H,particle_candidate)
#    b_T_mult_H_T = (matrix_H @ particle_candidate).T
#    value_candidate = -2*np.matmul(obs.T,K) + np.matmul(b_T_mult_H_T,K)
#    
#    
##    K = sigma_H*np.matmul(topos,particle_accepted)  
##    b_T_mult_H_T = np.matmul((particle_accepted).T,topos.T)
##    value_accepted = -2*np.matmul(obs.T,K) + np.matmul(b_T_mult_H_T,K)
#    K = sigma_H*np.matmul(matrix_H,particle_accepted)
#    b_T_mult_H_T = (matrix_H @ particle_accepted).T
#    value_accepted = -2*np.matmul(obs.T,K) + np.matmul(b_T_mult_H_T,K)
#    
#    #### likeli candidate
##    value_likeli_candi = np.exp(-(value_candidate)+(np.min([value_candidate,value_accepted])))
#    #### likeli current
##    value_likeli_curre = np.exp(-(value_accepted)+(np.min([value_candidate,value_accepted])))
#
#    ratio = np.exp(-0.5*(value_candidate-value_accepted))
#
#
##    if (value_likeli_curre==0) or (value_likeli_curre is np.inf):
##        ratio = 0
##    elif(value_likeli_curre is not np.inf) and (value_likeli_candi is np.inf):
##        ratio=1
##    else:
##        ratio = value_likeli_candi/value_likeli_curre
#    
#        
#    prob = np.min([1,ratio])
#    
#    
#    return prob
#    
#
# 
#    
#def mutation_MCMC_from_past_champ(pho,M,particle,noise,particle_past,delta_t,I,L,C, pchol_cov_noises,dt,obs,H,L_noise,topos,sigma_inverse,matrix_H):
#
#    particle_accepted =  particle
#    noise_accepted = noise
#    for iter_i in range(M):
#        
#        particle_candidate,noise_candidate = propagate_particle(pho,particle_past,noise_accepted,delta_t,I,L,C, pchol_cov_noises, dt)
#        prob_acept = calculate_acceptance_prob_champ(particle_candidate[:,0],particle_accepted,obs,H,L_noise,topos,sigma_inverse,matrix_H)
##        print(prob_acept)
#        if np.random.uniform(0,1)<prob_acept:
##            print(prob_acept)
#            particle_accepted = particle_candidate[:,0]
#            noise_accepted = noise_candidate
#    
#
#        
#        
#    return particle_accepted,noise_accepted    





def propagate_all_particles(particles_past_original,noises_accepted_original, dt, I,L,C,delta_t,pho,pchol_cov_noises,dt_adapted ):
    
#    bt_candidates = particles_past_original
#    noises_candidate_complete = np.zeros(shape = noises_accepted_original.shape)
#    
#    for iter_i in range(delta_t):
#        bt_candidates,noise_candidates = evol_forward_bt_MCMC(I,L,C, pchol_cov_noises, dt,bt_candidates,0,0,True,noises_accepted_original[iter_i,...],pho)
#        if iter_i==0:
#            noises_candidate_complete = noise_candidates
#        else: 
#            noises_candidate_complete = np.concatenate((noises_candidate_complete,noise_candidates),axis=0)
    
    bt_all_particles = particles_past_original # 6 x nb_particles
    noises_updated = np.zeros(shape=noises_accepted_original.shape) # time x 42 x nb_particles
    
    for time in range(delta_t-1):
        
        bt_all_particles,noise_time_n = evol_forward_bt_MCMC(I,L,C, pchol_cov_noises, dt,bt_all_particles,0,0,True,noises_accepted_original[time,...],pho)
        
        noises_updated[time,...] = noise_time_n
        
    
    time = delta_t-1
    bt_all_particles,noise_time_n = evol_forward_bt_MCMC(I,L,C, pchol_cov_noises,dt_adapted,bt_all_particles,0,0,True,noises_accepted_original[time,...],pho)
    noises_updated[time,...] = noise_time_n
    
    
    return bt_all_particles,noises_updated



def calculate_acceptance_prob_champ_all_particles_and_sample(particles_candidate,particles_accepted,obs_K,Hr_K,noises,noises_candidate):
    
    particles_candidate = np.vstack((particles_candidate,np.ones((1,particles_candidate.shape[1]))))
    particles_accepted = np.vstack((particles_accepted,np.ones((1,particles_accepted.shape[1]))))
    particles_chronos = np.concatenate((particles_accepted,particles_candidate),axis=1)
    
    
    
    values = (-2*obs_K @ particles_chronos + np.einsum('ij,ji->i', (particles_chronos.T @ Hr_K), particles_chronos))[0,:]
    
   
    
    
#    particles = np.concatenate((particles_accepted,particles_candidate),axis=1)
#    H_phi_b = np.matmul(matrix_H,particles)
#    a = np.einsum('ij,ji->i', H_phi_b.T, sigma_inverse*H_phi_b)
#    b = -2*np.matmul(obs.T,sigma_inverse*H_phi_b)[0,:]
#    value = b + a
    
    
#    H_phi_b = np.matmul(matrix_H,particles_candidate)
##    K = sigma_inverse*H_phi_b
##    b_T_mult_H_T = H_phi_b.T
##    del H_phi_b
##    a = np.matmul(b_T_mult_H_T,K)
#    a = np.einsum('ij,ji->i', H_phi_b.T, sigma_inverse*H_phi_b)
##    del b_T_mult_H_T
#    b = -2*np.matmul(obs.T,sigma_inverse*H_phi_b)[0,:]
##    del K
#    value_candidate = b + a
#    del a
#    del b
#    
#    
#    H_phi_b = np.matmul(matrix_H,particles_accepted)
##    K = sigma_inverse*H_phi_b
##    b_T_mult_H_T = H_phi_b.T
##    del H_phi_b
##    a = np.matmul(b_T_mult_H_T,K)
#    a = np.einsum('ij,ji->i', H_phi_b.T, sigma_inverse*H_phi_b)
##    del b_T_mult_H_T
#    b = -2*np.matmul(obs.T,sigma_inverse*H_phi_b)[0,:]
##    del K
#    value_accepted = b + a
#    del a
#    del b
#    
    value_accepted = values[:int(len(values)/2)]
    value_candidate = values[int(len(values)/2):]
    delta_value = value_candidate-value_accepted
    ratio = np.exp(-0.5*(delta_value))
    prob = np.minimum(np.ones(len(ratio)),ratio)
    
    
    particles_final = np.zeros((particles_candidate.shape[0]-1,particles_candidate.shape[1]))
    noise_final = np.zeros(noises.shape)
    for i in range(len(prob)):
        if np.random.uniform(0,1)<prob[i]:

            particles_final[:,i] = particles_candidate[:-1,i]
            noise_final[...,i] = noises_candidate[...,i]
            
        else:
            particles_final[:,i] = particles_accepted[:-1,i]
            noise_final[...,i] = noises[...,i]
            
    
    
    
    return particles_final,noise_final


def particle_filter(ILC_a_cst,obs,K,Hpiv_K,particles_chronos,N_threshold,noises,particles_past,nb_mutation_steps,dt_original,\
                    dt_adapted,pho,delta_t,pchol_cov_noises):
    
    
    I = ILC_a_cst['modal_dt']['I']
    L = ILC_a_cst['modal_dt']['L']
    C = ILC_a_cst['modal_dt']['C']

#    only_repeated = True
    
    
#    obs = create_virtual_obs(observation,lambda_values,beta_1)
    
#    if mask == True:
#         
#        matrix_H = matrix_H.reshape((observation.shape[0],observation.shape[1],observation.shape[2],observation.shape[3],matrix_H.shape[1]),order='F')
#        matrix_H = matrix_H[::subsampling_grid,::subsampling_grid,::subsampling_grid,:,:]
#        matrix_H = matrix_H.reshape((matrix_H.shape[0]*matrix_H.shape[1]*matrix_H.shape[2]*matrix_H.shape[3],matrix_H.shape[4]),order='F')
#        observation = observation[::subsampling_grid,::subsampling_grid,::subsampling_grid,:]
        
        
    ##### dim obs
#    x_size,y_size,z_size,dim = observation.shape
#    dimensions = {}
#    dimensions['x_size'] = x_size
#    dimensions['y_size'] = y_size
#    dimensions['z_size'] = z_size
#    dimensions['dim']    = dim
    
    
    
    
    
    
    
    
#    observation = np.reshape(observation,(int(observation.shape[0]*observation.shape[1]*observation.shape[2]*observation.shape[3])),order='F')
#    teste = topos @ np.concatenate((particles_chronos,np.ones(shape=(1,particles_chronos.shape[1]))))
#    error = np.sqrt(np.sum(np.power((teste - observation[...,np.newaxis]),2),axis=0))
#    print('Mean square error: '+str(error))
    
#    obs = create_virtual_champs_vitesse_DNS(observation,H,L_noise,dimensions,std_space,dX,subsampling_grid)[...,np.newaxis]
    
    
    
    
    obs_K = obs.T @ K
#    obs = observation[...,np.newaxis]
    
    
#    likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1,1)
    
    
    
    
    likelihood = calculate_champs_likelihood(particles_chronos,obs_K,1,Hpiv_K)
#    likelihood = np.ones(particles_chronos.shape[1])
    
    weigths = likelihood/np.sum(likelihood)
    ess = calculate_effective_sample_size(weigths)
    
#    if ess>N_threshold:
#        indexes = resample(weigths)
#        print('ESS: '+ str(ess))
#        ################################################################################
#        ###################################----MUTATION----#############################
#        if only_repeated ==True:
#            counter = collections.Counter(indexes)
#            keys = counter.keys()
#            indexes_repeated = []
#            for key in keys:
#                if counter[key]>0:
#                    indexes_repeated.append(key)
#        
#        particles_selected = np.zeros(shape=particles.shape)
#        noises_selected = np.zeros(shape=noises.shape)
#        i = 0
#        for index in indexes:
#            if index in indexes_repeated:
#                noise = noises[...,index]
#                particle = particles[:,index]
#                particle_past = particles_past[:,index]
#                particles_selected[...,i],noises_selected[...,i] = mutation_MCMC_from_past(pho,M,particle,noise,particle_past,delta_t,I,L,C, pchol_cov_noises,\
#                                                                                          dt,obs,lambda_values,beta_1)
#            else:
#                particles_selected[...,i] = particles[...,index]
#                noises_selected[...,i] = noises[...,index]
#                
#            i+=1
#            
#        
#        ############################################################################################
#        #################################---UPDATE---###############################################
#        particles = particles_selected
#        noises = noises_selected
#    
#        return particles,obs
    
    
    
    r = 0
    phi = 1
    phi_guime = 0
    
    while (r<10):
        
        
        if (phi_guime>0.95):
            index_print = np.where((weigths>0.05))
#            print('indexes with more than 5% probability of sampling: '+ str(index_print[0]))
#            print('phi guime: '+ str(phi_guime))
            
            phi = 1
            phi_guime = 1
#            print(len(collections.Counter(indexes).keys()))  
#            print('Number of temp iter: '+str(r))
            print('\n')
            
            return particles_chronos
        
        else:
#            likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1,1)
            if r>0:
                likelihood = calculate_champs_likelihood(particles_chronos,obs_K,1,Hpiv_K)
            phi_guime = find_tempering_coeff(likelihood,N_threshold,phi)
            if phi_guime>0.99:
                phi_guime = 0.99
                
            
            
            print('                    -- Tempering number '+str(r+1)+' --                ')
            print('phi guime: '+ str(phi_guime))
            

        print('phi effective:' + str(phi_guime-1+phi))
#        likelihood = calculate_likelihood(particles,obs,lambda_values,beta_1,phi_guime-1+phi)
        likelihood = calculate_champs_likelihood(particles_chronos,obs_K,phi_guime-1+phi,Hpiv_K)
#        likelihood = np.power(likelihood,(phi_guime-1+phi))
#        print(likelihood)
        weigths = (likelihood/np.sum(likelihood))[0,:]
        
        index_print = np.where((weigths>0.05))
        ess = calculate_effective_sample_size(weigths)
        print('indexes with more than 5% probability of sampling: '+ str(index_print[0]))
        print('ESS: '+ str(ess))

        indexes = resample(weigths)
        
        ################################################################################
        ###################################----MUTATION----#############################
#        if only_repeated ==True:
#            counter = collections.Counter(indexes)
#            keys = counter.keys()
#            indexes_repeated = []
#            for key in keys:
#                if counter[key]>0:
#                    indexes_repeated.append(key)
#        
#        particles_selected = np.zeros(shape=particles_chronos.shape)
#        noises_selected = np.zeros(shape=noises.shape)
#        i = 0
#        for index in indexes:
#            if index in indexes_repeated:
#                noise = noises[...,index]
#                particle = particles_chronos[:,index]
#                particle_past = particles_past[:,index]
##                particles_selected[...,i],noises_selected[...,i] = mutation_MCMC_from_past(pho,M,particle,noise,particle_past,delta_t,I,L,C, pchol_cov_noises,\
##                                                                                          dt,obs,lambda_values,beta_1)
#                
#                particles_selected[...,i],noises_selected[...,i] = mutation_MCMC_from_past_champ(pho,M,particle,noise,particle_past,delta_t,I,L,C, pchol_cov_noises,\
#                                                                                              dt,obs,H,L_noise,topos,sigma_inverse,matrix_H)
#                
#            else:
#                particles_selected[...,i] = particles_chronos[...,index]
#                noises_selected[...,i] = noises[...,index]
#                
#            i+=1
#            print('Mutation: ' +str(i)+' done')
        
        #### Resampling
        noises = noises[...,indexes]
        particles_chronos = particles_chronos[:,indexes]
        particles_past = particles_past[:,indexes]
        
        
        
        
        for mutation in range(nb_mutation_steps):
            particles_candidates,noises_candidate = propagate_all_particles(particles_past, noises, dt_original, I,L,C,delta_t,pho,pchol_cov_noises,dt_adapted )
##            calculate accept prob all and sample
            particles_chronos,noises= calculate_acceptance_prob_champ_all_particles_and_sample(particles_candidates,particles_chronos,obs_K,Hpiv_K,\
                                                                                                noises,noises_candidate)
            print('Mutation: '+str(mutation+1)+'/'+str(nb_mutation_steps))
        
        ############################################################################################
        #################################---UPDATE---###############################################
#        particles_chronos = particles_selected
#        noises = noises_selected
        
        
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
        
     
        print('-----------------------------------------------------------')
        
        
   
    
    print('\n')
    
    
    return particles_chronos
    
    


    
    
    
    
    
    
    
    









    
    
    
    
    