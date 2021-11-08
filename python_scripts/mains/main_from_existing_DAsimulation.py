# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:17:08 2019

@author: matheus.ladvig & valentin resseguier
"""

    
######################################----PARAMETERS TO CHOOSE----############################################
# Parameters choice
param_ref = {}
#param_ref['n_simu'] = 20   # 100      # Number of simulations steps in time
param_ref['n_simu'] = 100   # 100      # Number of simulations steps in time
#param_ref['N_particules'] = n_particles # Number of particles to select  
#    beta_1 = 0.1                            # beta_1 is the parameter that controls the noise to create virtual observation beta_1 * np.diag(np.sqrt(lambda))
beta_2 = 1.        # 1                    # beta_2 is the parameter that controls the  noise in the initialization of the filter
beta_3 = 1.                              # beta_3 is the parameter that controls the impact in the model noise -> beta_3 * pchol_cov_noises 
#    beta_4 = 5                              # beta_4 is the parameter that controls the time when we will use the filter to correct the particles
init_centred_on_ref = False              # If True, the initial condition is centered on the reference initial condiion
N_threshold = 40                        # Effective sample size in the particle filter
pho = 0.998                             # Constant that constrol the balance in the new brownian and the old brownian in particle filter
linewidth_ = 4.


#assimilate = 'real_data'                # The data that will be assimilated : 'real_data'  or 'fake_real_data' 
#nb_mutation_steps = 500  # 150 30                # Number of mutation steps in particle filter 
##nb_mutation_steps = 300  # 150 30                # Number of mutation steps in particle filter 

assimilate = 'fake_real_data'                # The data that will be assimilated : 'real_data'  or 'fake_real_data' 


#    L = 0.75*0.00254/(32*10**(-3))         # Incertitude associated with PIV data estimated before. It was used in the Sigma matrix estimation. 
std_space = 0.0065/(32*10**-3)          # Correlation distance in PIV measures
only_load = False                       # If False Hpiv*Topos will be calculated, if True and calculated before it will be loaded 
slicing = True                          # If True we will select one slice to assimilate data, because with 2d2c PIV we have only one slice.
slice_z = 30                            # The slice that will be assimilated: It should be 30 because the Hpiv*topos calculated in matlab take in account the slice 30
data_assimilate_dim = 2                 # In this experiments we assimilate 2D data, and in the case Reynolds=300, the vector flow in the z direction will be ignored.
u_inf_measured = 0.388                  # The PIV measured velocity (See the .png image in the respective PIV folder with all measured constants). It must be represented in m/s
cil_diameter = 12.                       # Cylinder diameter in PIV experiments. It must be in mm. (See the .png image in the respective PIV folder with all measured constants).
center_cil_grid_dns_x_index = 60        # Index that represents the center of the cylinder in X in the DNS grid
center_cil_grid_dns_y_index = 49        # Index that represents the center of the cylinder in Y in the DNS grid
#Re = 300                                # Reynolds constant
center_cil_grid_PIV_x_distance = -75.60 # Center of the cylinder in X in the PIV grid (See the .png image in the respective PIV folder with all measured constants). It ust be in mm.
center_cil_grid_PIV_y_distance = 0.75   # Center of the cylinder in Y in the PIV grid (See the .png image in the respective PIV folder with all measured constants). It ust be in mm.

#SECONDS_OF_SIMU = 70. #70. #0.5                    # We have 331 seconds of real PIV data for reynolds=300 beacuse we have 4103 files. --> ( 4103*0.080833 = 331).....78 max in the case of fake_PIV
sub_sampling_PIV_data_temporaly = True  # True                                                           # We can choose not assimilate all possible moments(time constraints or filter performance constraints or benchmark constraints or decorraltion hypotheses). Hence, select True if subsampling necessary 
## factor_of_PIV_time_subsampling_gl = 10  
#factor_of_PIV_time_subsampling_gl = int(5 / 0.080833)                                                           # The factor that we will take to subsampled PIV data. 


plt_real_time = False                                                                     # It can be chosen to plot chronos evolution in real time or only at the end of the simulation
plot_period = 2 * float(5/10)/2
heavy_real_time_plot = True # Compute confidence interval for real-time plots
#n_frame_plots = 20           
fig_width= 9
fig_height = 4      
plot_Q_crit = False
plot_ref_gl = True


mask_obs = True      # True            # Activate spatial mask in the observed data

## Case 6 
#subsampling_PIV_grid_factor_gl = 3   # 1     # Subsampling constant that will be applied in the observed data, i.e if 3 we will take 1 point in 3 
#x0_index_gl = 10  # 10                                                                                           # Parameter necessary to chose the grid that we will observe(i.e if 6 we will start the select the start of the observed grid in the 6th x index, hence we will reduce the observed grid).
#nbPoints_x_gl = 1      # 70    nbPoints_x <= (202 - x0_index) /subsampling_PIV_grid_factor                  # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
#y0_index_gl = 50         # 10                                                                                   # Parameter necessary to chose the grid that we will observe(i.e if 30 we will start the observed grid in the 30th y index, hence we will reduce the observed grid).
#nbPoints_y_gl = 1      # 30   nbPoints_y <= (74 - y0_index) /subsampling_PIV_grid_factor                       # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
##dt_PIV = 0.080833
##factor_of_PIV_time_subsampling_gl = int(5/10 / dt_PIV) 
#assimilation_period = float(5/10)

# Case 1 
subsampling_PIV_grid_factor_gl = 3   # 1     # Subsampling constant that will be applied in the observed data, i.e if 3 we will take 1 point in 3 
x0_index_gl = 10  # 10                                                                                           # Parameter necessary to chose the grid that we will observe(i.e if 6 we will start the select the start of the observed grid in the 6th x index, hence we will reduce the observed grid).
nbPoints_x_gl = 1      # 70    nbPoints_x <= (202 - x0_index) /subsampling_PIV_grid_factor                  # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
y0_index_gl = 10         # 10                                                                                   # Parameter necessary to chose the grid that we will observe(i.e if 30 we will start the observed grid in the 30th y index, hence we will reduce the observed grid).
nbPoints_y_gl = 1      # 30   nbPoints_y <= (74 - y0_index) /subsampling_PIV_grid_factor                       # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
#dt_PIV = 0.080833
#factor_of_PIV_time_subsampling_gl = int(5/10 / dt_PIV) 
assimilation_period = float(5/10)

## Case 2
#subsampling_PIV_grid_factor_gl = 3   # 1     # Subsampling constant that will be applied in the observed data, i.e if 3 we will take 1 point in 3 
#x0_index_gl = 10  # 10                                                                                           # Parameter necessary to chose the grid that we will observe(i.e if 6 we will start the select the start of the observed grid in the 6th x index, hence we will reduce the observed grid).
#nbPoints_x_gl = 3     # 70    nbPoints_x <= (202 - x0_index) /subsampling_PIV_grid_factor                  # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
#y0_index_gl = 10         # 10                                                                                   # Parameter necessary to chose the grid that we will observe(i.e if 30 we will start the observed grid in the 30th y index, hence we will reduce the observed grid).
#nbPoints_y_gl = 3     # 30   nbPoints_y <= (74 - y0_index) /subsampling_PIV_grid_factor                       # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
##factor_of_PIV_time_subsampling_gl = int(5 / 0.080833) 
#assimilation_period = float(5)

## Case 3
#subsampling_PIV_grid_factor_gl = 10   # 1     # Subsampling constant that will be applied in the observed data, i.e if 3 we will take 1 point in 3 
#x0_index_gl = 10  # 10                                                                                           # Parameter necessary to chose the grid that we will observe(i.e if 6 we will start the select the start of the observed grid in the 6th x index, hence we will reduce the observed grid).
#nbPoints_x_gl = 3      # 70    nbPoints_x <= (202 - x0_index) /subsampling_PIV_grid_factor                  # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
#y0_index_gl = 10         # 10                                                                                   # Parameter necessary to chose the grid that we will observe(i.e if 30 we will start the observed grid in the 30th y index, hence we will reduce the observed grid).
#nbPoints_y_gl = 3      # 30   nbPoints_y <= (74 - y0_index) /subsampling_PIV_grid_factor                       # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
##factor_of_PIV_time_subsampling_gl = int(5 / 0.080833) 
#assimilation_period = float(5/10) 

### Case 4
#subsampling_PIV_grid_factor_gl = 10   # 1     # Subsampling constant that will be applied in the observed data, i.e if 3 we will take 1 point in 3 
#x0_index_gl = 10  # 10                                                                                           # Parameter necessary to chose the grid that we will observe(i.e if 6 we will start the select the start of the observed grid in the 6th x index, hence we will reduce the observed grid).
#nbPoints_x_gl = 3      # 70    nbPoints_x <= (202 - x0_index) /subsampling_PIV_grid_factor                  # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
#y0_index_gl = 10         # 10                                                                                   # Parameter necessary to chose the grid that we will observe(i.e if 30 we will start the observed grid in the 30th y index, hence we will reduce the observed grid).
#nbPoints_y_gl = 3      # 30   nbPoints_y <= (74 - y0_index) /subsampling_PIV_grid_factor                       # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
##factor_of_PIV_time_subsampling_gl = int(5 / 0.080833) 
#assimilation_period = float(5/20) 

### Case 5
#subsampling_PIV_grid_factor_gl = 3   # 1     # Subsampling constant that will be applied in the observed data, i.e if 3 we will take 1 point in 3 
#x0_index_gl = 10  # 10                                                                                           # Parameter necessary to chose the grid that we will observe(i.e if 6 we will start the select the start of the observed grid in the 6th x index, hence we will reduce the observed grid).
#nbPoints_x_gl = 10      # 70    nbPoints_x <= (202 - x0_index) /subsampling_PIV_grid_factor                  # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
#y0_index_gl = 10         # 10                                                                                   # Parameter necessary to chose the grid that we will observe(i.e if 30 we will start the observed grid in the 30th y index, hence we will reduce the observed grid).
#nbPoints_y_gl = 10      # 30   nbPoints_y <= (74 - y0_index) /subsampling_PIV_grid_factor                       # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
##factor_of_PIV_time_subsampling_gl = int(5 / 0.080833) 
#assimilation_period = float(5/10) 

## Case full
#subsampling_PIV_grid_factor_gl = 3   # 1     # Subsampling constant that will be applied in the observed data, i.e if 3 we will take 1 point in 3 
#x0_index_gl = 0  # 10                                                                                           # Parameter necessary to chose the grid that we will observe(i.e if 6 we will start the select the start of the observed grid in the 6th x index, hence we will reduce the observed grid).
#nbPoints_x_gl = 67      # 70    nbPoints_x <= (202 - x0_index) /subsampling_PIV_grid_factor                  # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
#y0_index_gl = 0         # 10                                                                                   # Parameter necessary to chose the grid that we will observe(i.e if 30 we will start the observed grid in the 30th y index, hence we will reduce the observed grid).
#nbPoints_y_gl = 24      # 30   nbPoints_y <= (74 - y0_index) /subsampling_PIV_grid_factor                       # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
##factor_of_PIV_time_subsampling_gl = int(5 / 0.080833) 
#assimilation_period = float(5/10) 
        
color_mean_EV = 'deepskyblue'
color_quantile_EV = 'paleturquoise'
        
plot_debug = False
pos_Mes = -7

#import matplotlib.pyplot as plt
import math
import os
import shutil
from convert_mat_to_python import convert_mat_to_python
from convert_mat_to_python import open_struct
from convert_mat_to_python_EV import convert_mat_to_python_EV
from pathlib import Path
import sys
import hdf5storage
import numpy as np
import scipy.io as sio
path_functions = Path(__file__).parents[1].joinpath('functions')
sys.path.insert(0, str(path_functions))
from fct_cut_frequency_2_full_sto import fct_cut_frequency_2_full_sto
from evol_forward_bt_RK4 import evol_forward_bt_RK4
from evol_forward_bt_MCMC import evol_forward_bt_MCMC
from fct_name_2nd_result import fct_name_2nd_result
from particle_filter import particle_filter
import matplotlib.pyplot as plt
from scipy import interpolate
#from scipy import sparse as svds
import scipy.sparse as sps
from PIL import Image
import time as t_exe
import json 
from plot_bt_dB_MCMC_varying_error import plot_bt_dB_MCMC_varying_error_DA
    
    
#%%                                          Begin the main_from_existing_ROM that constrols all the simulation
    
def main_from_existing_DAsimulation(nb_modes,threshold,type_data,nb_period_test,\
                           no_subampl_in_forecast,reconstruction,\
                           adv_corrected,modal_dt,n_particles,test_fct,svd_pchol,\
                                               stochastic_integration,\
                                               estim_rmv_fv,eq_proj_div_free,\
                                               thrDtCorrect,\
                                               noBugSubsampl,\
                                               choice_n_subsample,EV,\
                                               nb_mutation_steps,\
                                               SECONDS_OF_SIMU):#nb_modes,threshold,type_data,nb_period_test,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt):
        
    param_ref['N_particules'] = n_particles # Number of particles to select  
        
    if not mask_obs:   # If we must select a smaller grid inside the observed grid. 
        x0_index = 1.
        y0_index = 1.
        nbPoints_x = float('nan')
        nbPoints_y = float('nan')
        subsampling_PIV_grid_factor = 1
    else:
        x0_index = x0_index_gl
        y0_index = y0_index_gl
        nbPoints_x = nbPoints_x_gl
        nbPoints_y = nbPoints_y_gl
        subsampling_PIV_grid_factor = subsampling_PIV_grid_factor_gl
    
    
    switcher = {
    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated': 300 ,
    'DNS100_inc3d_2D_2018_11_16_blocks_truncated': 100  
    }
    Re = switcher.get(type_data,[float('Nan')])
    
    if assimilate == 'real_data':
        switcher = {
        'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated': float(0.080833),
        'DNS100_inc3d_2D_2018_11_16_blocks_truncated' : float(0.05625)
        }
        dt_PIV = switcher.get(type_data,[float('Nan')])
        if not sub_sampling_PIV_data_temporaly:
            factor_of_PIV_time_subsampling = 1
        else:
            factor_of_PIV_time_subsampling = int(assimilation_period/dt_PIV)
        plot_ref = plot_ref_gl 
#        dt_PIV = 0.080833                                                                                     # Temporal step between 2 consecutive PIV images. (See the .png image in the respective PIV folder with all measured constants).    
        number_of_PIV_files = int(SECONDS_OF_SIMU/dt_PIV) + 1                                                 # Number of PIV files to load
        vector_of_assimilation_time = np.arange(start=0,stop=number_of_PIV_files*dt_PIV,step=dt_PIV) # Construct the moments that can be assimilated.
#        vector_of_assimilation_time = np.arange(start=dt_PIV,stop=(number_of_PIV_files+1)*dt_PIV,step=dt_PIV) # Construct the moments that can be assimilated.
        vector_of_assimilation_time = vector_of_assimilation_time[::factor_of_PIV_time_subsampling]              # Using the factor to select the moments that we will take to assimilate
    elif assimilate == 'fake_real_data':
        plot_ref = plot_ref_gl                     # Plot bt_tot
        switcher = {
        'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated': 80 ,
        'DNS100_inc3d_2D_2018_11_16_blocks_truncated': 14  
        }
        nb_file_learning_basis = switcher.get(type_data,[float('Nan')])
        switcher = {
        'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated': 0.25,
        'DNS100_inc3d_2D_2018_11_16_blocks_truncated' : 0.05
        }
        dt_PIV = switcher.get(type_data,[float('Nan')])
        if not sub_sampling_PIV_data_temporaly:
            factor_of_PIV_time_subsampling = 1
        else:
            factor_of_PIV_time_subsampling = int(assimilation_period/dt_PIV)
    
    #%%  Parameters already chosen 
    
    current_pwd = Path(__file__).parents[1] # Select the path
    folder_results = current_pwd.parents[0].joinpath('resultats').joinpath('current_results') # Select the current results path
    folder_data = current_pwd.parents[1].joinpath('data') # Select the data path
#    folder_data = current_pwd.parents[0].joinpath('data') # Select the data path
    
    
    param_ref['folder_results'] = str(folder_results) # Stock folder results path
    param_ref['folder_data'] = str(folder_data)       # Stock folder data path
    
    
    modal_dt_ref = modal_dt # Define modal_dt_ref
    
    
    
    #%% Get data
    
    ############################ Construct the path to select the model constants I,L,C,pchol and etc.
    
    file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_'  \
              + choice_n_subsample
    if choice_n_subsample == 'auto_shanon' :
        file = file + '_threshold_' + str(threshold)
    file = file + test_fct   
    file = file + '_fullsto' # File where the ROM coefficients are save
    file = file + '/'
    if not adv_corrected:
        file = file + '_no_correct_drift'
    if thrDtCorrect:
        file = file + '_thrDtCorrect'      
    file = file + '_integ_' + stochastic_integration
    if estim_rmv_fv:
        file = file + '_estim_rmv_fv'
    if noBugSubsampl:
        file = file + '_noBugSubsampl'      
    if eq_proj_div_free == 2:
        file = file + '_DFSPN'            
#    file_save = file
    file = file + '.mat'
    file_res = folder_results / Path(file)

    # The function creates a dictionary with the same structure as the Matlab Struct in the path file_res
    I_sto,L_sto,C_sto,I_deter,L_deter,C_deter,plot_bts,pchol_cov_noises,bt_tot,param = convert_mat_to_python(str(file_res)) # Call the function and load the matlab data calculated before in matlab scripts.
    param['decor_by_subsampl']['no_subampl_in_forecast'] = no_subampl_in_forecast                                           # Define the constant
    param['dt'] = float(param['dt'])
    
    # Remove subsampling effect
    param['dt'] = param['dt'] /param['decor_by_subsampl']['n_subsampl_decor']
    param['N_test'] = param['N_test'] * param['decor_by_subsampl']['n_subsampl_decor']
    param['N_tot'] = param['N_test'] + 1
    param['decor_by_subsampl']['n_subsampl_decor'] = 1
    
    if EV:
        file_EV= 'EV_result_' + type_data + '_' + str(nb_modes) + '_modes'
        file_EV= file_EV + '_noise.mat'
        file_EV_res = folder_results / Path(file_EV)
        ILC_EV = convert_mat_to_python_EV(str(file_EV_res)) # Call the function and load the matlab data calculated before in matlab scripts.

    
    
    #%% Redefined path to get acces to data
    
    param['nb_period_test'] = nb_period_test
    param['decor_by_subsampl']['test_fct'] = test_fct
    folder_data = param_ref['folder_data']
    folder_results = param_ref['folder_results']
    
    big_data = False
    
    param['folder_data'] = str(folder_data)
    param['folder_results'] = str(folder_results)
    param['big_data'] = big_data
    param['plots_bts'] = plot_bts
    
    
    param['folder_results'] = param_ref['folder_results']
    param['N_particules'] = param_ref['N_particules']
    n_simu = param_ref['n_simu']
#    param['N_tot'] = bt_tot.shape[0]
#    param['N_test'] = param['N_tot'] - 1
#    bt_tot = bt_tot[:param['N_test'] + 1,:]                # Ref. Chronos in the DNS cas
#    time_bt_tot = np.arange(0,bt_tot.shape[0],1)*param['dt']
#    bt_tronc=bt_tot[0,:][np.newaxis]                       # Define the initial condition as the reference
    
    lambda_values = param['lambda'] # Define the constant lambda (The integral in one period of the square temporal modes )

        
    #%% Folder to save data assimilation plot results
    plt.close('all')
##    file_plots = '3rd' + file_save[3:] + '/'
#    file_plots = '3rdresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
#            choice_n_subsample + '_threshold_' + str(threshold) + test_fct + '/'
    file_plots = '3rdresult/' + type_data + '_' + str(nb_modes) + '_modes_'  \
              + choice_n_subsample
    if choice_n_subsample == 'auto_shanon' :
        file_plots = file_plots + '_threshold_' + str(threshold)
    file_plots = file_plots + test_fct + '/'
    if modal_dt:
        file_plots = file_plots + '_modal_dt'    
    if not adv_corrected:
        file_plots = file_plots + '_no_correct_drift'
#    file_plots = file_plots + '_integ_Ito'
    if thrDtCorrect:
        file_plots = file_plots + '_thrDtCorrect'
    file_plots = file_plots + '_integ_' + stochastic_integration
    if estim_rmv_fv:
        file_plots = file_plots + '_estim_rmv_fv'
    if svd_pchol==1:
        file_plots = file_plots + '_svd_pchol'
    elif svd_pchol==2:
        file_plots = file_plots + '_svd_pchol2'
    if noBugSubsampl:
        file_plots = file_plots + '_noBugSubsampl'
    if eq_proj_div_free == 2:
        file_plots = file_plots + '_DFSPN'       
    file_plots = file_plots + '/' + assimilate + \
                              '/_DADuration_' + str(int(SECONDS_OF_SIMU)) + '_'
    if sub_sampling_PIV_data_temporaly:
        file_plots = file_plots + 'ObsSubt_' + str(int(factor_of_PIV_time_subsampling)) + '_'    
    if mask_obs:
        file_plots = file_plots + 'ObsMaskyy_sub_' + str(int(subsampling_PIV_grid_factor)) \
                 + '_from_' + str(int(x0_index)) + '_to_' \
                 + str(int(x0_index + nbPoints_x*subsampling_PIV_grid_factor)) \
                 + '_from_' + str(int(y0_index)) + '_to_' \
                 + str(int(y0_index+nbPoints_y*subsampling_PIV_grid_factor)) + '_'
    else:
        file_plots = file_plots + 'no_mask_'
    if init_centred_on_ref:
        file_plots = file_plots + 'initOnRef_'
    file_plots = file_plots + 'beta_2_' + str(int(beta_2))
    file_plots = file_plots + '_nSimu_' + str(int(n_simu))
    file_plots = file_plots + '_nMut_' + str(int(nb_mutation_steps))
    file_plots = file_plots + '_nPcl_' + str(int(n_particles))
        
#    file_plots = file_plots.replace(".", "_")
    folder_results_plot = os.path.dirname(os.path.dirname(os.path.dirname(folder_results)))
    file_plots_res = os.path.join(folder_results_plot, file_plots) 
    file_res = file_plots_res / Path('chronos.mat')
    
#    dict_python = {}
#    sio.loadmat(file_res,dict_python)
    
    dict_python = hdf5storage.loadmat(str(file_res))
    # The param has the key 'param'
    param_ = dict_python['param']
    # Create the dictionary that will stock the data
    param = {}
    
    param = open_struct(param_)
    
##    param = dict_python['param']
#    param['N_test']=dict_python['param']['N_test']
#    param['N_tot']=dict_python['param']['N_tot']
#    param['dt'] = dict_python['param']['dt']
    time_bt_tot = dict_python['time_bt_tot']
    bt_tot = dict_python['bt_tot']
    quantiles_PIV = dict_python['quantiles_bt_tot']
    dt_PIV = dict_python['dt_PIV']
    index_pf = dict_python['index_pf']
#    np.array(time)[np.array(index_pf)[1:]] = dict_python['time_DA']
    Re = dict_python['Re']
    time = dict_python['time']
    particles_mean = dict_python['particles_mean']
    particles_median = dict_python['particles_median']
    bt_MCMC = dict_python['bt_MCMC']
    quantiles = dict_python['quantiles']
    if EV:
        particles_mean_EV = dict_python['particles_mean_EV']
        particles_median_EV = dict_python['particles_median_EV']
        bt_forecast_EV = dict_python['bt_forecast_EV']
        quantiles_EV = dict_python['quantiles_EV']
    
    #%% PLOTSSSSSSSSSSSS
    
    ##############################################################################################################
    #################################---TEST PLOTS---#############################################################
    dt_tot = param['dt']
    N_test = param['N_test'] 
    
    time_bt_tot = time_bt_tot[0,:]
    time = time[0,:]
    
    particles_mean = np.mean(bt_MCMC[:,:,:],axis=2)
    particles_median = np.median(bt_MCMC[:,:,:],axis=2)
    quantiles = np.quantile(bt_MCMC[:,:,:],q=[0.025,0.975],axis=2)
    if EV:
        particles_mean_EV = np.mean(bt_forecast_EV[:,:,:],axis=2)
        particles_median_EV = np.median(bt_forecast_EV[:,:,:],axis=2)
        quantiles_EV = np.quantile(bt_forecast_EV[:,:,:],q=[0.025,0.975],axis=2)
    n_particles = bt_MCMC.shape[-1] 


    for index in range(particles_mean.shape[1]):
        plt.figure(index,figsize=(12, 9))
        plt.ylim(-10, 10)
        ####        delta = 1.96*particles_std_estimate[:,index]/np.sqrt(n_particles)
        if EV:
            plt.fill_between(time,quantiles_EV[0,:,index],quantiles_EV[1,:,index],color=color_quantile_EV)
            line1_EV = plt.plot(time,particles_mean_EV[:,index],'-', \
                                color=color_mean_EV, label = 'EV particles mean',linewidth=linewidth_)
        if plot_ref==True:
            if assimilate == 'real_data':
                plt.plot(time_bt_tot,quantiles_PIV[0,:,index],'k--',label = 'True state',linewidth=linewidth_)
                plt.plot(time_bt_tot,quantiles_PIV[1,:,index],'k--',label = 'True state',linewidth=linewidth_)
            else:
                plt.plot(time_bt_tot,bt_tot[:,index],'k--',label = 'True state',linewidth=linewidth_)
            
            
        plt.fill_between(time,quantiles[0,:,index],quantiles[1,:,index],color='gray')
            
        line1 = plt.plot(time,particles_mean[:,index],'b-',label = 'Red LUM particles mean',linewidth=linewidth_)
#            if EV:
#                line1_EV = plt.plot(time,particles_mean_EV[:,index],'-', \
#                                    color=color_mean_EV, label = 'EV particles mean')
            
#        line2 = plt.plot(time_bt_tot,ref[:,index],'k--',label = 'True state')
#        line2 = plt.plot(time_bt_tot,ref[:,index],'k--',label = 'True state')
#        line3 = plt.plot(time_simu,particles_median[:,index],'g.',label = 'particles median')
#        line4 = plt.plot(dt_tot*np.concatenate((np.zeros((1)),np.array(time_pf))),particles_estimate[:,index],'m.',label = 'PF mean estimation')
        plt.plot(np.array(time)[np.array(index_pf)[1:]],pos_Mes*np.ones((len(index_pf[1:]))),'r.',linewidth=linewidth_)
        plt.grid()
        plt.ylabel('Chronos '+r'$b'+str(index+1)+'$'+' amplitude',fontsize=20)
        plt.xlabel('Time',fontsize=20)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        file_res_mode = file_plots_res / Path(str(index+1) + '.pdf')
        plt.savefig(file_res_mode,dpi=200 )
#    
    
    param['truncated_error2'] = param['truncated_error2'][...,np.newaxis]


    if EV:
        N_ = particles_mean.shape[0]
        struct_bt_MCMC = {}
        struct_bt_MCMC['mean'] = particles_mean\
            .copy()[:N_:n_simu]
        struct_bt_MCMC['var'] = np.var(bt_MCMC[:,:,:],axis=2)\
            .copy()[:N_:n_simu]
        struct_bt_MEV_noise = {}
        struct_bt_MEV_noise['mean'] = particles_mean_EV\
            .copy()[:N_:n_simu]
        struct_bt_MEV_noise['var'] = np.var(bt_forecast_EV[:,:,:],axis=2)\
            .copy()[:N_:n_simu]
        param['N_test']=int(param['N_test']/n_simu)
        param['N_tot']=param['N_test']+1
        param['dt'] = param['dt'] * n_simu
        plot_bt_dB_MCMC_varying_error_DA(file_plots_res, \
                param, bt_tot, struct_bt_MEV_noise, struct_bt_MCMC)
        
    del C_deter 
    del C_sto 
    del L_deter 
    del L_sto 
    del I_deter 
    del I_sto

    return 0 #var
    
    #%%
    
    
    
    
    
    
    