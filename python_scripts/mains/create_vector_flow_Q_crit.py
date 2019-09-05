# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:56:40 2019

@author: matheus.ladvig
"""

import numpy as np
import hdf5storage
from pathlib import Path
import scipy.io as sio
import json 


if __name__ == '__main__':
    n_modes = 6
    
    
    file = 'tensor_mode_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated_'+str(n_modes)+'_modes'
#    file = 'mode_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated_'+str(n_modes)+'_modes'
    name_file_data = Path(__file__).parents[3].joinpath('data').joinpath(file)
    mat = hdf5storage.loadmat(str(name_file_data))
    Omega_phi_m_U = mat['Omega_phi_m_U']
    S_phi_m_U = mat['S_phi_m_U']
#    phi_m_U = mat['phi_m_U']
    
    
    file = str(n_modes) + 'modes_particle_mean_Numpy'
    name_file_data = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath(file)
    mat = hdf5storage.loadmat(str(name_file_data))
    MX = mat['MX'][0]
    particles_mean = mat['particles_mean']
    
    for i in range(5):
        particles_mean = mat['particles_mean']
        particles_mean = np.hstack((particles_mean,np.ones((particles_mean.shape[0],1))))[i,:]
        particles_mean = np.tile(particles_mean,([Omega_phi_m_U.shape[0],Omega_phi_m_U.shape[2],Omega_phi_m_U.shape[3],1]))
        particles_mean = np.transpose(particles_mean,(0,3,1,2))
        
        Omega = np.multiply(particles_mean,Omega_phi_m_U)
        Omega = np.sum(Omega,axis=1)
        Omega = np.sum(np.sum(np.power(Omega,2),axis=2),axis=1)
        
        S = np.multiply(particles_mean,S_phi_m_U)
        S = np.sum(S,axis=1)
        S = np.sum(np.sum(np.power(S,2),axis=2),axis=1)
        
        Q = 0.5 * ( Omega - S )
        del Omega
        del S
        
        Q = np.reshape(Q,(MX))
        
        file = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath('aurore')
        name_file_data = file.joinpath('sequence_teste_Q'+str(i)+'.json')
        
        
        
        with open(str(name_file_data), 'w') as f:
            json.dump(Q.tolist(), f)
    
    
    
    
   
    
    
    
    
    
    
    
   

    
    
    
    

    