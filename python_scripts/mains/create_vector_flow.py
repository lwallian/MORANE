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
    
    
    file = 'mode_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated_'+str(n_modes)+'_modes'
    name_file_data = Path(__file__).parents[3].joinpath('data').joinpath(file)
    mat = hdf5storage.loadmat(str(name_file_data))
    phi_m_U = mat['phi_m_U']
    
    
    file = str(n_modes) + 'modes_particle_mean_Numpy'
    name_file_data = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath(file)
    mat = hdf5storage.loadmat(str(name_file_data))
    MX = mat['MX'][0]
    particles_mean = mat['particles_mean']
    
    for i in range(5):
        particles_mean = mat['particles_mean']
        particles_mean = np.hstack((particles_mean,np.ones((particles_mean.shape[0],1))))[i,:]
        tile = np.tile(particles_mean,([phi_m_U.shape[0],phi_m_U.shape[2],1]))
        particles_mean = np.transpose(tile,(0,2,1))
        
        multip = np.multiply(particles_mean,phi_m_U)
        sum_multip = np.sum(multip,axis=1)
        flow = np.sqrt(np.sum(np.power(sum_multip,2),axis=1))
        
        flow = np.reshape(flow,(MX))
        
        
    
        file = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath('aurore')
        name_file_data = file.joinpath('sequence_teste'+str(i)+'.json')
        
        
        
        with open(str(name_file_data), 'w') as f:
            json.dump(flow.tolist(), f)
    
    
    
    
   
    
    
    
    
    
    
    
   

    
    
    
    

    