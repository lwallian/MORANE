# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:54:16 2019

@author: matheus.ladvig
"""

import numpy as np
import math
from main_from_existing_ROM import main_from_existing_ROM

#def return_variances(nb_modes,threshold,type_data,nb_period_test,\
#                     no_subampl_in_forecast,reconstruction,\
#                     adv_corrected,modal_dt,nb_particle):
#    
#    
#    return 2
    
    
    
if __name__ == '__main__':
    vect_nb_modes = [2,4,6,8,16]
    type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
    nb_period_test = math.nan
    nb_modes_max = np.max(vect_nb_modes)
    modal_dt = True
    threshold = 1e-06
    adv_corrected = False
    reconstruction =  False
    no_subampl_in_forecast = False      
    min_particles = 10
    max_particles = 100
    nb_particles = [10,100,1000]
    
    
    
#    variances = np.zeros((len(vect_nb_modes),len(nb_particles)))   
    variance_local = np.zeros((1,len(nb_particles)))    
    for j,nb_modes in enumerate(vect_nb_modes):              
        for i,nb_particle in enumerate(nb_particles):
            var = main_from_existing_ROM(nb_modes,threshold,type_data,nb_period_test,\
                                         no_subampl_in_forecast,reconstruction,\
                                         adv_corrected,modal_dt,nb_particle)
            
            variance_local[0,i] = var
            print('Mode:'+ str(nb_modes))
            print('Number of particles: '+str(nb_particle))
            print('\n')
            
        np.save('variances_'+str(min_particles)+'_to_'+str(max_particles)+'_'+str(nb_modes)+'modes',variance_local)  
#        variances[j,:] = variance_local
        variance_local = np.zeros((1,len(nb_particles)))
        
      
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    