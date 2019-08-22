# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:13:50 2019

@author: matheus.ladvig
.


"""
import numpy as np
import math
from main_from_existing_ROM import main_from_existing_ROM
from super_main_from_existing_ROM_Simulation import super_main_from_existing_ROM_Simulation
#import matplotlib.pyplot as plt
#def super_main_from_existing_ROM(vect_nb_modes,type_data,v_threshold,vect_modal_dt,\
#    no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected):
#    pass
#    
    
def switch_type_data(argument):
    switcher = {
        'incompact3D_noisy2D_40dt_subsampl_truncated': [[1e-05],[False]],
        'DNS100_inc3d_2D_2018_11_16_blocks_truncated': [[1e-06],[True]], # BEST
        'turb2D_blocks_truncated': [[1e-05],[False,True]],
        'incompact3d_wake_episode3_cut_truncated': [[1e-06],[False]],
        'incompact3d_wake_episode3_cut': [[1e-06],[False]],
        'LES_3D_tot_sub_sample_blurred': [[1e-03],[True]],
        'inc3D_Re3900_blocks': [[1e-03],[True]],
        'inc3D_Re3900_blocks_truncated': [[1e-03],[True]],
        'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated':[[1e-04],[True]] # BEST
       
    }
    return switcher.get(argument,[[0.0005],[False]])
    
    
if __name__ == '__main__':

#    vect_nb_modes = [16,8,6,4,2]
    vect_nb_modes = [6] # Select the number of solved temporal modes
    no_subampl_in_forecast = False 
    vect_reconstruction = [False] # for the super_main_from_existing_ROM
    vect_adv_corrected = [False]
    test_fct = 'b'
    svd_pchol = True
    choice_n_subsample = 'auto_shanon'
#                           DATASET 
#    type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated'  #dataset to debug
    type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated' # Reynolds 300
#    type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'  # Reynolds 100
#%%   
#Important parameters    
#    Threshold used in the estimation of the optimal subsampling time step
#     if modal-dt = true,
#          (mimic the use of a) disctinct subsampling time step for the
#           differentials equations of distincts chronos
    
    # Get threshold and modal_dt
    v_threshold,vect_modal_dt = switch_type_data(type_data)
    
    nb_period_test = math.nan
    nb_modes_max = np.max(vect_nb_modes)
#    variances = []
    for modal_dt in vect_modal_dt:
        for threshold in v_threshold:
            for adv_corrected in vect_adv_corrected:
                for reconstruction in vect_reconstruction:
                    for k in vect_nb_modes:
                        n_particle = 100
                        main_from_existing_ROM(k,threshold,type_data,nb_period_test,\
                                               no_subampl_in_forecast,reconstruction,\
                                               adv_corrected,modal_dt,n_particle,\
                                               test_fct,svd_pchol,choice_n_subsample)
    

    
    


          
            
            
            
            
#    super_main_from_existing_ROM_Simulation(vect_nb_modes,type_data,v_threshold,vect_modal_dt,\
#                                             no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected)
#            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    

