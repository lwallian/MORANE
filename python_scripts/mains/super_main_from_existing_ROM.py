# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:13:50 2019

@author: matheus.ladvig, valentin resseguier
.


"""
import numpy as np
import math
import time
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

    vect_nb_modes = [2,4,6,8]
#    vect_nb_modes = [8,6,4,2]
    no_subampl_in_forecast = False 
    vect_reconstruction = [False] # for the super_main_from_existing_ROM
    vect_adv_corrected = [True]
    test_fct = 'b'
    svd_pchol = 2
    choice_n_subsample = 'htgen2'
#    choice_n_subsample = 'auto_shanon'
    stochastic_integration = 'Ito'
    estim_rmv_fv = True
#    eq_proj_div_free = 1
#    vect_eq_proj_div_free = 2
    vect_eq_proj_div_free = [2]
    EV = True
    thrDtCorrect = False
    noBugSubsampl = False
    
    #nb_mutation_steps = 0                # Number of mutation steps in particle filter 
    vect_nb_mutation_steps = [-1]                # Number of mutation steps in particle filter 
#    vect_nb_mutation_steps = [30,0]                # Number of mutation steps in particle filter 


#                           DATASET 
#    type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated'  #dataset to debug
    
    type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated' # Reynolds 300
    SECONDS_OF_SIMU = 70. #70. #0.5                    # We have 331 seconds of real PIV data for reynolds=300 beacuse we have 4103 files. --> ( 4103*0.080833 = 331).....78 max in the case of fake_PIV

#    type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'  # Reynolds 100
#    SECONDS_OF_SIMU = 100. #70. #0.5                   
    
#%%   
#Important parameters    
#    Threshold used in the estimation of the optimal subsampling time step
#     if modal-dt = true,
#          (mimic the use of a) disctinct subsampling time step for the
#           differentials equations of distincts chronos
    
    # Get threshold and modal_dt
    if choice_n_subsample == 'auto_shanon' :
        v_threshold,vect_modal_dt = switch_type_data(type_data)
    else:
         v_threshold = [float('nan')]
         vect_modal_dt = [False]
#    if not( choice_n_subsample == 'auto_shanon'):
#        vect_modal_dt = [False]
    
    nb_period_test = math.nan
    nb_modes_max = np.max(vect_nb_modes)
#    variances = []
    for eq_proj_div_free in vect_eq_proj_div_free:
        for nb_mutation_steps in vect_nb_mutation_steps:
            for modal_dt in vect_modal_dt:
                for threshold in v_threshold:
                    for adv_corrected in vect_adv_corrected:
                        for reconstruction in vect_reconstruction:
                            for k in vect_nb_modes:
                                n_particle = 100
                                main_from_existing_ROM(k,threshold,type_data,nb_period_test,\
                                                       no_subampl_in_forecast,reconstruction,\
                                                       adv_corrected,modal_dt,n_particle,\
                                                       test_fct,svd_pchol,\
                                                       stochastic_integration,\
                                                       estim_rmv_fv,eq_proj_div_free,\
                                                       thrDtCorrect,\
                                                       noBugSubsampl,\
                                                       choice_n_subsample,EV,\
                                                       nb_mutation_steps,
                                                       SECONDS_OF_SIMU)
    

    
    


          
            
            
            
            
#    super_main_from_existing_ROM_Simulation(vect_nb_modes,type_data,v_threshold,vect_modal_dt,\
#                                             no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected)
#            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    

