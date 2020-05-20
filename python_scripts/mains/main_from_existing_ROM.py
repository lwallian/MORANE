# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:17:08 2019

@author: matheus.ladvig
"""

    
######################################----PARAMETERS TO CHOOSE----############################################
# Parameters choice
param_ref = {}
param_ref['n_simu'] = 100   # 100      # Number of simulations steps in time
#param_ref['N_particules'] = n_particles # Number of particles to select  
#    beta_1 = 0.1                            # beta_1 is the parameter that controls the noise to create virtual observation beta_1 * np.diag(np.sqrt(lambda))
beta_2 = 1.        # 1                    # beta_2 is the parameter that controls the  noise in the initialization of the filter
beta_3 = 1.                              # beta_3 is the parameter that controls the impact in the model noise -> beta_3 * pchol_cov_noises 
#    beta_4 = 5                              # beta_4 is the parameter that controls the time when we will use the filter to correct the particles
init_centred_on_ref = False              # If True, the initial condition is centered on the reference initial condiion
N_threshold = 40                        # Effective sample size in the particle filter
pho = 0.998                             # Constant that constrol the balance in the new brownian and the old brownian in particle filter


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


plt_real_time = True                                                                     # It can be chosen to plot chronos evolution in real time or only at the end of the simulation
plot_period = 2 * float(5/10)/2
heavy_real_time_plot = True # Compute confidence interval for real-time plots
#n_frame_plots = 20           
fig_width= 9
fig_height = 4      
plot_Q_crit = True
plot_ref_gl = False


mask_obs = True      # True            # Activate spatial mask in the observed data

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


def calculate_rotational(topos_Fx,topos_Fy,delta_space,nb_x,nb_y):
    
    '''
    We need to calculate the z curl component of the fluid
    '''
    
    Fx = np.reshape(topos_Fx,(nb_x,nb_y,topos_Fx.shape[1]),order='F')  # Reshape the vector in x direction
    Fy = np.reshape(topos_Fy,(nb_x,nb_y,topos_Fy.shape[1]),order='F')  # Reshape the vector in y direction
    
    Field_dFy_dx = np.zeros((Fy.shape[0]-2,Fy.shape[1],Fy.shape[2]))   # Create vector to stock dFy/dx
    
    Field_dFx_dy = np.zeros((Fx.shape[0],Fx.shape[1]-2,Fx.shape[2]))   # Create vector to stock dFx/dy 
    
    
    for component in range(Fy.shape[-1]):                                                          # Get Vector in y direction for all components(number of modes if Topos)
#        print(component)
        for line in range(nb_y):                                                                   # Go through all the lines calculating the derivative 
            i=0
            for point in range(1,nb_x-1):                                                          # For each line calculate the derivative in all y
                dFy_dx = (Fy[point+1,line,component]-Fy[point-1,line,component])/(2*delta_space)   # Derivative
                Field_dFy_dx[i,line,component] = dFy_dx                                            # Stocks 
                i = i+1
                
    for component in range(Fx.shape[-1]):                                                             # Get Vector in y direction for all components(number of modes if Topos)
#        print(component)
        for collum in range(nb_x):                                                                    # Go through all the lines taking the derivative 
            i=0
            for point in range(1,nb_y-1):                                                             # For each collumn calculate the derivative in all x
                dFx_dy = (Fx[collum,point+1,component]-Fx[collum,point-1,component])/(2*delta_space)  # Derivative
                Field_dFx_dy[collum,i,component] = dFx_dy                                             # Stocks
                i = i+1
    
    curl_Z = Field_dFy_dx[:,1:-1,:] - Field_dFx_dy[1:-1,...]   # Take the rotational
    
    return curl_Z
    
    
    
#%%                                          Begin the main_from_existing_ROM that constrols all the simulation
    
def main_from_existing_ROM(nb_modes,threshold,type_data,nb_period_test,\
                           no_subampl_in_forecast,reconstruction,\
                           adv_corrected,modal_dt,n_particles,test_fct,svd_pchol,\
                                               stochastic_integration,\
                                               estim_rmv_fv,eq_proj_div_free,\
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
#        dt_PIV = 0.25                       # Temporal step between 2 consecutive PIV images.
#        nb_snapshots_each_file = 16         # The amount of snapshots that each file contains
#        time_per_file = (nb_snapshots_each_file)*dt_PIV
#        if SECONDS_OF_SIMU%time_per_file == 0:
#            number_of_FAKE_PIV_files = SECONDS_OF_SIMU/time_per_file
#        else:
#            number_of_FAKE_PIV_files = int(SECONDS_OF_SIMU/time_per_file) + 1
    
    #%%  Initialize randm generator
    np.random.seed()      
    
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
#        file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
#            choice_n_subsample + '_threshold_' + str(threshold) + test_fct   
#        file = file_save
    file = file + '/'
    if not adv_corrected:
        file = file + '_no_correct_drift'
    file = file + '_integ_' + stochastic_integration
    if estim_rmv_fv:
        file = file + '_estim_rmv_fv'
    if eq_proj_div_free == 2:
        file = file + '_DFSPN'            
#    file_save = file
    file = file + '.mat'
    file_res = folder_results / Path(file)

#    # On which function the Shanon ctriterion is used
##    test_fct = 'b'  # 'b' is better than db
#    a_t = '_a_cst_' 
#
#    file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
#            a_t + '_decor_by_subsampl_bt_decor_choice_' + choice_n_subsample 
#    if choice_n_subsample == 'auto_shanon' :
#        file = file + '_threshold_' + str(threshold)
#    file = file +'fct_test_' + test_fct    
##    file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
##            a_t + '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' + str(threshold) + \
##            'fct_test_' + test_fct    
##    var_exits =  'var' in locals() or 'var' in globals()
##    period_estim = 'period_estim' in locals() or 'period_estim' in globals()
##    if var_exits == True and period_estim == True:
##        file = file + '_p_estim_' + str(period_estim);
#    file = file + '_fullsto' # File where the ROM coefficients are save
#    print(file)
##    file_save = file
#    if not adv_corrected:
#        file = file + '_no_correct_drift'
#    file = file + '.mat'
#    file_res = folder_results / Path(file)
#    if not os.path.exists(file_res):
#        file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_'  \
#                  + choice_n_subsample
#        if choice_n_subsample == 'auto_shanon' :
#            file = file + '_threshold_' + str(threshold)
#        file = file + test_fct   
#        file = file + '_fullsto' # File where the ROM coefficients are save
##        file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
##            choice_n_subsample + '_threshold_' + str(threshold) + test_fct   
##        file = file_save
#        if not adv_corrected:
#            file = file + '/_no_correct_drift'
#        file_save = file
#        file = file + '.mat'
#        file_res = folder_results / Path(file)
#        if not os.path.exists(file_res):
#            file = file_save + '_integ_Ito'
#            file = file + '.mat'
#            file_res = folder_results / Path(file)
#    print(file)
    
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

        
    #%% Reduction of the noise matrix
    if svd_pchol:
        sq_lambda = np.sqrt(lambda_values)
        
        pchol_cov_noises_add = pchol_cov_noises[range(nb_modes),:]
#        pchol_cov_noises_add = pchol_cov_noises_add[np.newaxis,:,:]
        pchol_cov_noises = pchol_cov_noises[nb_modes:,:]
        pchol_cov_noises= np.reshape(pchol_cov_noises, \
                    (nb_modes,nb_modes,(nb_modes+1)*nb_modes),order='F') 
        
        for j in range(nb_modes):
            pchol_cov_noises_add[j,:] = (1/sq_lambda[j]) * \
                                                pchol_cov_noises_add[j,:]
        for i in range(nb_modes):
            for j in range(nb_modes):
                pchol_cov_noises[i,j,:] = (sq_lambda[i]/sq_lambda[j]) \
                                               * pchol_cov_noises[i,j,:]
                                               
        pchol_cov_noises= np.reshape(pchol_cov_noises, \
                    (nb_modes*nb_modes,(nb_modes+1)*nb_modes),order='F') 
        pchol_cov_noises = np.concatenate( \
                    (pchol_cov_noises_add,pchol_cov_noises),axis=0)
        

        U_cov_noises, S_cov_noises, _ = sps.linalg.svds(pchol_cov_noises, k=nb_modes)
        pchol_cov_noises = U_cov_noises @ np.diag(S_cov_noises)
        
        pchol_cov_noises_add = pchol_cov_noises[range(nb_modes),:]
        pchol_cov_noises = pchol_cov_noises[nb_modes:,:]
        pchol_cov_noises= np.reshape(pchol_cov_noises, \
                    (nb_modes,nb_modes,nb_modes),order='F') 
        
        for i in range(nb_modes):
            for j in range(nb_modes):
                pchol_cov_noises[i,j,:] = (1/(sq_lambda[i]/sq_lambda[j])) \
                                               * pchol_cov_noises[i,j,:]
        for j in range(nb_modes):
            pchol_cov_noises_add[j,:] = (1/(1/sq_lambda[j])) * \
                                                pchol_cov_noises_add[j,:]
##        pchol_cov_noises_add = pchol_cov_noises[0,:,:]
#        pchol_cov_noises_add= np.reshape(pchol_cov_noises_add, \
#                    (nb_modes,nb_modes),order='F') 
#        pchol_cov_noises = pchol_cov_noises[1:,:,:]
        pchol_cov_noises= np.reshape(pchol_cov_noises, \
                    (nb_modes*nb_modes,nb_modes),order='F') 
        pchol_cov_noises = np.concatenate( \
                    (pchol_cov_noises_add,pchol_cov_noises),axis=0)
        
#        if EV:
#            U_cov_noises_EV, S_cov_noises_EV, _ = \
#            sps.linalg.svds(ILC_EV['pchol_cov_noises'], k=nb_modes)
#            ILC_EV['pchol_cov_noises'] = U_cov_noises_EV @ np.diag(S_cov_noises_EV)

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
    file_plots = file_plots + '_integ_' + stochastic_integration
    if estim_rmv_fv:
        file_plots = file_plots + '_estim_rmv_fv'
    if svd_pchol:
        file_plots = file_plots + '_svd_pchol'
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
        
#    file_plots = file_plots.replace(".", "_")
    folder_results_plot = os.path.dirname(os.path.dirname(os.path.dirname(folder_results)))
    file_plots_res = os.path.join(folder_results_plot, file_plots) 
#    file_plots_res = folder_results / Path(file_plots)
#    print( file_plots_res )
    if not os.path.exists(file_plots_res):
        os.makedirs( file_plots_res )
    
    if plot_Q_crit:
        # File to save Q cirterion for real time 3D plots
        path_Q_crit = Path(__file__).parents[3].\
            joinpath('data_after_filtering').joinpath('Q_RedLUM')
#            joinpath('data_after_filtering').joinpath('aurore')
        if os.path.exists(path_Q_crit):
            shutil.rmtree(path_Q_crit)
            plt.pause(1)
        os.makedirs( path_Q_crit )
#        if EV:
#            # File to save Q cirterion for real time 3D plots
#            path_Q_crit_EV = Path(__file__).parents[3].\
#                joinpath('data_after_filtering').joinpath('Q_EV')
#            if os.path.exists(path_Q_crit_EV):
#                shutil.rmtree(path_Q_crit_EV)
#            os.makedirs( path_Q_crit_EV )
#        if plot_ref:
#            # File to save Q cirterion for real time 3D plots
#            path_Q_crit_ref = Path(__file__).parents[3].\
#                joinpath('data_after_filtering').joinpath('Q_ref')
#            if os.path.exists(path_Q_crit_ref):
#                shutil.rmtree(path_Q_crit_ref)
#            os.makedirs( path_Q_crit_ref )
#        if not os.path.exists(path_Q_crit):
#            os.makedirs( path_Q_crit )
        if EV:
            # File to save Q cirterion for real time 3D plots
            path_Q_crit_EV = Path(__file__).parents[3].\
                joinpath('data_after_filtering').joinpath('Q_EV')
            if not os.path.exists(path_Q_crit_EV):
                os.makedirs( path_Q_crit_EV )
        if plot_ref:
            # File to save Q cirterion for real time 3D plots
            path_Q_crit_ref = Path(__file__).parents[3].\
                joinpath('data_after_filtering').joinpath('Q_ref')
            if not os.path.exists(path_Q_crit_ref):
                os.makedirs( path_Q_crit_ref )
            
    
    #%% Parameters of the ODE of the b(t)
    
    modal_dt = modal_dt_ref
    
    tot = {'I':I_sto,'L':L_sto,'C':C_sto}
    
    I_sto = I_sto - I_deter
    L_sto = L_sto - L_deter
    C_sto = C_sto - C_deter
    
    deter = {'I':I_deter,'L':L_deter,'C':C_deter}
    
    sto = {'I':I_sto,'L':L_sto,'C':C_sto}
    
    ILC = {'deter':deter,'sto':sto,'tot':tot}
    
    ILC_a_cst = ILC.copy()
    
    #%% Choice of modal time step
   
    ##############################################################################
    if modal_dt == True:
#        rate_dt,ILC_a_cst,pchol_cov_noises = fct_cut_frequency_2_full_sto(bt_tot,ILC_a_cst,param,pchol_cov_noises,modal_dt)
        rate_dt, ILC_a_cst,pchol_cov_noises = fct_cut_frequency_2_full_sto(bt_tot,ILC_a_cst,param,pchol_cov_noises,modal_dt)
        ILC_a_cst = ILC_a_cst['modal_dt']
    else:
        ILC_a_cst = ILC_a_cst['tot']
#        ILC_a_cst['modal_dt'] = ILC_a_cst['tot']
    
    
    #%% Do not temporally subsample, in order to prevent aliasing in the results
    N_tot_max = int(SECONDS_OF_SIMU/param['dt'])+1
    N_tot = param['N_tot']
    param['N_tot'] = N_tot
    param['N_test'] = param['N_tot'] - 1
    if N_tot > N_tot_max:
        N_tot = N_tot_max
    if not reconstruction:
        if assimilate == 'fake_real_data':
    #        if exists: 
            current_pwd = Path(__file__).parents[1]
#            if param['nb_period_test'] is math.nan:
#                name_file_data = current_pwd.parents[1].joinpath('data').joinpath( type_data + '_' + str(nb_modes) + '_modes' + '_threshold_' + str(param['decor_by_subsampl']['spectrum_threshold']) + \
#                                                   '_nb_period_test_' + 'NaN' + '_Chronos_test_basis.mat')
#            else:
#                name_file_data = current_pwd.parents[1].joinpath('data').joinpath( type_data + '_' + str(nb_modes) + '_modes' + '_threshold_' + str(param['decor_by_subsampl']['spectrum_threshold']) + \
#                                                   '_nb_period_test_' + str(param['nb_period_test']) + '_Chronos_test_basis.mat')
            name_file_data = current_pwd.parents[1].joinpath('data').\
            joinpath( type_data + '_' + str(nb_modes) + '_modes' + \
                     '_subsample_1_nb_period_test_NaN_Chronos_test_basis.mat')               
           
                
            if name_file_data.exists():
                mat = hdf5storage.loadmat(str(name_file_data))
                bt_tot = mat['bt']
                truncated_error2 = mat['truncated_error2']
                param['truncated_error2'] = truncated_error2
                dt_bt_tot = param['dt']/param['decor_by_subsampl']['n_subsampl_decor']
                
                
                
            else:
                print('ERROR: File does not exist ', str(name_file_data))
                return 0
            
            if param['big_data'] == True:
                print('Test basis creation done')
            
            #Test basis creation
            
#            param['N_tot'] = bt_tot.shape[0]
            param['N_tot'] = N_tot
            param['N_test'] = param['N_tot'] - 1
            bt_tot = bt_tot[:int(param['N_test']\
                    *param['decor_by_subsampl']['n_subsampl_decor'] + 1),:]                # Ref. Chronos in the DNS cas
#            bt_tot = bt_tot[:(param['N_test'] + 1),:]                # Ref. Chronos in the DNS cas
            time_bt_tot = np.arange(0,bt_tot.shape[0],1)*dt_bt_tot
            if len(time_bt_tot.shape)>1:
                time_bt_tot = time_bt_tot[0,:]
#            time_bt_tot = np.arange(0,bt_tot.shape[0],1)*param['dt']
#            bt_tronc=bt_tot[0,:][np.newaxis]                       # Define the initial condition as the reference
            quantiles_PIV = np.zeros((2,bt_tot.shape[0],bt_tot.shape[1]))
        else:
            file = (Path(__file__).parents[3]).joinpath('data_PIV').\
            joinpath('bt_tot_PIV_Re'+str(int(Re))+'_n'+str(nb_modes)+'.mat')
#            joinpath('bt_tot_PIV_Re'+str(int(Re))+'.mat')
            print(file)
            dict_python = hdf5storage.loadmat(str(file))
            bt_tot = dict_python['bt_tot_PIV']
            quantiles_PIV = dict_python['quantiles_PIV']
            dt_PIV = dict_python['dt_PIV']
            N_tot_PIV_max = int(SECONDS_OF_SIMU/dt_PIV)+1
            if bt_tot.shape[0] > N_tot_PIV_max-1:
                bt_tot = bt_tot[:N_tot_PIV_max,:]
                quantiles_PIV = quantiles_PIV[:,:N_tot_PIV_max,:]
            time_bt_tot = np.arange(0,bt_tot.shape[0],1)*dt_PIV
            time_bt_tot = time_bt_tot[0,:]
#            bt_tot = float('nan')
#            truncated_error2 = float('nan')
            truncated_error2 = np.zeros((bt_tot.shape[0],bt_tot.shape[1]))
            param['truncated_error2'] = truncated_error2
#            bt_tronc = float('nan')
    
    bt_tronc=bt_tot[0,:][np.newaxis]                       # Define the initial condition as the reference
        
    
    #%% Time integration of the reconstructed Chronos b(t)
    
#    Reconstruction in the deterministic case
#    bt_forecast_deter = bt_tronc
#    for index in range(param['N_test']):
#        bt_forecast_deter = np.vstack((bt_forecast_deter, evol_forward_bt_RK4(ILC_a_cst['deter']['I'],ILC_a_cst['deter']['L'],ILC_a_cst['deter']['C'],param['dt'],bt_forecast_deter)))
    
#    Reconstruction in the stochastic case
#    bt_forecast_sto = bt_tronc
#    for index in range(param['N_test']):
#        bt_forecast_sto = np.vstack((bt_forecast_sto,evol_forward_bt_RK4(ILC_a_cst['modal_dt']['I'],ILC_a_cst['modal_dt']['L'],ILC_a_cst['modal_dt']['C'],param['dt'],bt_forecast_sto)))
    
    param['dt'] = param['dt']/n_simu                       # The simulation time step is dependent of the number of time evolution steps between the param['dt'],therefore the new param['dt'] is divided by the number of evolution steps 
    param['N_test'] = param['N_test'] * n_simu             # Number of model integration steps is now the number of steps before times the number of integration steps between two old steps
    
    
    
    
#    Reconstruction in the stochastic case
  

    
    if init_centred_on_ref:
        bt_MCMC = np.tile(bt_tronc.T,(1,1,param['N_particules']))   # initializes the chronos of all particules equally
#        shape = (1,bt_tronc.shape[1],int(param['N_particules']))    # Define the shape of this vector containing all the chronos of all particles
#        bt_MCMC[:,:,:] =  bt_MCMC + \
#        beta_2*np.tile(np.sqrt(lambda_values)[...,np.newaxis],(1,1,bt_MCMC[:,:,:].shape[2]))*np.random.normal(0,1,size=shape)
##        beta_2*np.tile(lambda_values[...,np.newaxis],(1,1,bt_MCMC[:,:,:].shape[2]))*np.random.normal(0,1,size=shape)
    else:
        bt_MCMC = np.zeros((1,nb_modes,param['N_particules']))
    
    bt_MCMC[:,:,:] = bt_MCMC + \
        beta_2*np.tile(np.sqrt(lambda_values)[...,np.newaxis],(1,1,param['N_particules']))\
            *np.random.normal(0,1,size=(1,nb_modes,param['N_particules'])) # Initialise the chronos paricles randomly and dependent of the lambda values 
#        beta_2*np.tile(np.sqrt(lambda_values)[...,np.newaxis],(1,1,bt_MCMC[:,:,:].shape[2]))*np.random.normal(0,1,size=shape)
    # Initialise the chronos particles randomly and dependent of the lambda values 

#    bt_fv = bt_MCMC.copy()                                              # Define bt_fv 
#    bt_m = np.zeros((1,int(param['nb_modes']),param['N_particules']))   # Define bt_m
    iii_realization = np.zeros((param['N_particules'],1))               # Define iii_realization that is necessary if any explosion in the simulation
    if EV:
        bt_forecast_EV = bt_MCMC.copy()
        iii_realization_EV = iii_realization.copy()
    
        
   #%%  Loading matrices H_PivTopos calculated before, select the data in the new PIV grid and load sigma_inverse in the PIV space
   
   
#      LOAD TOPOS
    print('Loading H_PIV @ Topos...')
    path_topos = Path(folder_data).parents[0].joinpath('data_PIV').\
            joinpath('mode_'+type_data+'_'+str(nb_modes)+'_modes_PIV') # Topos path 
    topos_data = hdf5storage.loadmat(str(path_topos))                                                                 # Load topos
    topos = topos_data['phi_m_U']   
    MX_PIV = topos_data['MX_PIV'].astype(int)
    MX_PIV = tuple(map(tuple,MX_PIV))[0]
    coordinates_x_PIV= topos_data['x_PIV_after_crop']
    coordinates_y_PIV= topos_data['y_PIV_after_crop']
    coordinates_x_PIV= np.reshape(coordinates_x_PIV,MX_PIV,order='F') 
    coordinates_y_PIV= np.reshape(coordinates_y_PIV,MX_PIV,order='F') 
    coordinates_x_PIV =  np.transpose(coordinates_x_PIV[:,0])  
    coordinates_y_PIV = coordinates_y_PIV[0,:]                                            # Select Topos          
#    MX = param['MX']  
   
##      LOAD TOPOS
#    print('Loading Topos...')
#    path_topos = Path(folder_data).parents[1].joinpath('data').joinpath('mode_'+type_data+'_'+str(nb_modes)+'_modes') # Topos path 
#    topos_data = hdf5storage.loadmat(str(path_topos))                                                                 # Load topos
#    topos = topos_data['phi_m_U']                                                                                     # Select Topos          
#    MX = param['MX'][0,:]                                                                                             # Select DNS grid
#    
#    if topos.shape[-1]==3:                        # Select only the u and v vector field.
#        topos = topos[...,:data_assimilate_dim]   # Selecting...
#    
    dim = topos.shape[-1]                                                                              # Define the vectorial field dimension
    topos = np.transpose(topos,(0,2,1))                                                              # Rearrange dimensions 
#    topos_l = np.transpose(topos,(0,2,1))                                                              # Rearrange dimensions 
#    topos_l = np.reshape(topos_l,(int(topos_l.shape[0]*topos_l.shape[1]),topos_l.shape[2]),order='F')  # Reshape the topos, the last dimensions being the number of resolved modes plus one and the first dimensions is (Nx * Ny * dim)

    grid = param['MX']                                                                     # Define the DNS grid
    distance = param['dX']                                                                     # Define the spatial space between 2 samples in DNS grid 
#    grid = param['MX'][0]                                                                              # Define the DNS grid
#    distance = param['dX'][0,0]                                                                        # Define the spatial space between 2 samples in DNS grid 

#    matrix_H,number = calculate_H_PIV(topos_l,distance,grid,std_space,only_load,dim,slicing,slice_z)   # Apply spatial filter in the topos. The filtr was estimated before and is based in the PIV measures
#    
#    print('\nCalculating PIV mask and applying on the Topos...')
#    matrix_H = np.transpose(matrix_H,(0,1,2,4,3)) 
##    imgplot = plt.imshow(matrix_H)                                                                                                                               # Rearrange  matrix_H
#    matrix_H = np.reshape(matrix_H,(int(matrix_H.shape[0]*matrix_H.shape[1]*matrix_H.shape[2]),matrix_H.shape[3],matrix_H.shape[4]),order='F')                                   # Reshape matrix_H  
#    if assimilate in ['real_data','fake_real_data']:                                                                                                                                                # Select if real data to transform Hpiv_Topos from space DNS to PIV
#        topos_new_coordinates,coordinates_x_PIV,coordinates_y_PIV = reduce_and_interpolate_topos_same_as_matlab(matrix_H,param['grid'][0],MX,data_assimilate_dim,slicing,\
#                                                                                                             slice_z,u_inf_measured,cil_diameter,center_cil_grid_dns_x_index,\
#                                                                                                             center_cil_grid_dns_y_index,center_cil_grid_PIV_x_distance,\
#                                                                                                            center_cil_grid_PIV_y_distance,std_space)
#    else:
#        print('Error: Data to assimilate is not known.')
#        sys.exit()
              
    '''
    The Sigma_inverse matrix was calculated before in the space H_piv, so we need just to load it.
    
        - The folder ''data_PIV'' contains all files related to measured data 
        - Therefore we need to search HSigSigH_PIV..... and load it here. It was calculated before in matlab.
        
                                         ---------------------------VERY IMPORTANT--------------------------------
                                         
            - THIS MATRIX IS DEPENDENT OF THE PIV MEASURED NOISE. THEREFORE THE MATRIX L. BEING SIGMA = [HSigSigH + LL]; 
            - THE VALUE USED HERE WAS ESTIMATED IN THE IRSTEA DATA AND REPRESENTS 6% OF THE AMPLITUDE.
            - THE NOISE IS UNCORRELATED IN TIME AND IN SPACE. SUBSAMPLING SPATIALLY AND TEMPORALLY CAN INCREASE THE POSSIBILITY OF BE TRUE.
    
    '''
    threshold_ = str(threshold).replace('.', '_',)
    path_Sigma_inverse = Path(__file__).parents[3].joinpath('data_PIV').\
    joinpath('HSigSigH_PIV_'+type_data+'_'+str(nb_modes)\
             +'_modes_a_cst_threshold_'+ threshold_)  # Load Sigma_inverse
#    path_Sigma_inverse = Path(__file__).parents[3].joinpath('data_PIV').joinpath('HSigSigH_PIV_'+type_data+'_'+str(param['nb_modes'])+'_modes_a_cst_threshold_0_'+str(threshold)[2:])  # Load Sigma_inverse
    Sigma_inverse_data = hdf5storage.loadmat(str(path_Sigma_inverse)) # Select Sigma_inverse
    
    
    '''
    - The subgrid that Valentin used in the Matlab code was not exactly the original PIV grid. Therefore we have not found the same grid. Therefore, the indexes 3->-4 and 1->-1 cutting 'coordinates_x_PIV' and 
    'coordinates_y_PIV' transforms my estimated grid in the same grid estimated by valentin.
    '''
#    coordinates_x_PIV = coordinates_x_PIV[3:-4]                            # Transforming the grid in x 
#    coordinates_y_PIV = coordinates_y_PIV[1:-1]                            # Transforming the grid in y
#    topos_new_coordinates = topos_new_coordinates[3:-4,1:-1,:,:]           # Selecting the data in valentin grid
#    topos_new_coordinates = np.transpose(topos_new_coordinates,(0,1,3,2))  # Transposing the topos 
    topos_new_coordinates = np.reshape(topos,\
                              MX_PIV + tuple(np.array([data_assimilate_dim,(nb_modes+1)])),order='F') 
#                              (len(coordinates_x_PIV),len(coordinates_y_PIV),\
#                               data_assimilate_dim,(nb_modes+1)),order='F') 
    
    
    # Plots for debug
    if plot_debug:
        matrix_H_plot = topos_new_coordinates.copy()
    #    matrix_H_plot = np.reshape(matrix_H_plot,(len(coordinates_x_PIV),len(coordinates_y_PIV),matrix_H.shape[1],matrix_H.shape[2]),order='F') 
        fig=plt.figure(1)
        imgplot = plt.imshow(np.transpose(matrix_H_plot[:,:,0,-1]),\
                             interpolation='none', extent=\
                             [coordinates_x_PIV[0],coordinates_x_PIV[-1],\
                              coordinates_y_PIV[0],coordinates_y_PIV[-1]])
        plt.pause(1) 
        fig.colorbar(imgplot)
        plt.pause(1) 
    
    
    Hpiv_Topos = np.reshape(topos_new_coordinates,(int(topos_new_coordinates.shape[0]*topos_new_coordinates.shape[1]*topos_new_coordinates.shape[2]),topos_new_coordinates.shape[3]),order='F') # The topos that we have estimated reshaped to posterior matrix multiplications

#    np.save('Hpiv_Topos.npy',Hpiv_Topos) # If save, it will save the topos in a numpy file in the same folder of the script.
    
    #%%  Define and apply mask in the observation
    '''
    In the lines below we define and apply the observation mask. It allows define where we'll observe in the PIV grid and
    if necessary with smaller spatial grid. 
    
    Matrices that should receive M_mask:
        - Y_obs
        - H_piv_Topos
        - Sigma_inverse
    
    The information about the grid points in x and y spatially represented are stocked in 
    coordinates_x_PIV and coordinates_y_PIV, respectivelly.
    
    ''' 
    if mask_obs == True:   # If we must select a smaller grid inside the observed grid. 
        
        xEnd_index = x0_index + nbPoints_x*subsampling_PIV_grid_factor  # Define the last x index of the new grid
        yEnd_index = y0_index + nbPoints_y*subsampling_PIV_grid_factor  # Define the last y new grid index
#        print(len(coordinates_y_PIV-1))
#        print(len(coordinates_x_PIV-1))
        if (yEnd_index-subsampling_PIV_grid_factor)>(len(coordinates_y_PIV-1)) or  (xEnd_index-subsampling_PIV_grid_factor)>(len(coordinates_x_PIV-1)): # Checks if the points are inside the observed grid
            print('Error: grid selected is bigger than the observed')
            sys.exit()
        
        coordinates_x_PIV_with_MASK = coordinates_x_PIV[x0_index:xEnd_index:subsampling_PIV_grid_factor]  # Selecting the new grid 
        coordinates_y_PIV_with_MASK = coordinates_y_PIV[y0_index:yEnd_index:subsampling_PIV_grid_factor]  # Selecting the new grid
        
        '''
        We need to construct the mask to select the coefficients in the matrices.
        '''
        Mask_x = np.zeros(len(coordinates_x_PIV),dtype = np.int8)    # Construct the mask x
        Mask_x[x0_index:xEnd_index:subsampling_PIV_grid_factor] = 1  # Define the points of the new grid as 1
        
        Mask_y = np.zeros(len(coordinates_y_PIV),dtype = np.int8)    # Construct the mask y
        Mask_y[y0_index:yEnd_index:subsampling_PIV_grid_factor] = 1  # Define the points of the new grid as 1
        
        ###############################--Begin the mask construction--#############
        if Mask_y[0]==0:                                       # If the first collum dont  belongs to the grid
            Mask_final = np.zeros(len(Mask_x),dtype = np.int8) # Add a zeros to the mask
        else:                                                  # If it belongs
            Mask_final = Mask_x.copy()                         # Add the Mask x 
        
        for line in Mask_y[1:]:                                                                     # The process continues to y>1 until the last column
            if line==0:
                Mask_final = np.concatenate((Mask_final,np.zeros(len(Mask_x),dtype = np.int8)))
            else: 
                Mask_final = np.concatenate((Mask_final,Mask_x.copy()))
                
                
#        Mask_final = np.concatenate((Mask_final,Mask_final))                                        # It must be concatenated with himself because we are working with 2 dimensions
#        Mask_final_bool = Mask_final.astype(bool)                                                   # Transform the data inside in boolean. 1->True and 0->False
#    
#        print('The coordinates that will be observed: ')
#        print('The x coordinates: '+str(coordinates_x_PIV_with_MASK))
#        print('The y coordinates: '+str(coordinates_y_PIV_with_MASK))
    else:
        Mask_final = np.ones(int(len(coordinates_x_PIV)*len(coordinates_y_PIV)),dtype = np.int8)    # Construct the mask
        coordinates_x_PIV_with_MASK = coordinates_x_PIV
        coordinates_y_PIV_with_MASK = coordinates_y_PIV
#        x0_index = 1
#        y0_index = 1
        xEnd_index = len(coordinates_x_PIV)
        yEnd_index = len(coordinates_y_PIV)
#        subsampling_PIV_grid_factor = 1
        
    Mask_final = np.concatenate((Mask_final,Mask_final))                                        # It must be concatenated with himself because we are working with 2 dimensions
    Mask_final_bool = Mask_final.astype(bool)                                                   # Transform the data inside in boolean. 1->True and 0->False
    
    print('The coordinates that will be observed: ')
    print('The x coordinates: '+str(coordinates_x_PIV_with_MASK))
    print('The y coordinates: '+str(coordinates_y_PIV_with_MASK))
#    
    #%%   Calculate Sigma_inverse
    
    print('Loading Sigma and multipling likelihood matrices')
    Sigma_inverse = Sigma_inverse_data['inv_HSigSigH'][:,0,:,:]                             # Load Sigma inverse 
    Sigma_inverse = Sigma_inverse[Mask_final_bool[:Sigma_inverse.shape[0]],:,:].copy()      # SelectSigma inverse in the mask that we observe    
    
    ##### Transform this matrix in a square matrix
    nb_points = Sigma_inverse.shape[0]                                                      # Number of points in the grid
    nb_dim = Sigma_inverse.shape[1]                                                         # Dimension
    
    
    ######## Applying final mask after define observation window
    Hpiv_Topos = Hpiv_Topos[Mask_final_bool,:].copy()        
    
    
    '''
    The sigma inversed must be transformed in a matrix 
        - It contains the inversed matrix of correlations that is uncorrelated in space and in time. The correlation is between dimensions. 
        
    '''
#    Sigma_inverse_squared = np.zeros((int(nb_points*nb_dim),int(nb_points*nb_dim)))     # Create a matrix to stock the data     
##    print(nb_points)
##    print(nb_dim)
#    for line in range(int(nb_points)):                                                  # To all spatial samples we create the first part of the matrix that contains the correlation of Vx
#        Sigma_inverse_squared[line,line] = Sigma_inverse[line,0,0]                      # Correlation of Vx with Vx
#        Sigma_inverse_squared[line,line+nb_points] =  Sigma_inverse[line,0,1]           # Correlation of Vx with Vy 
#       
#    for line in range(int(nb_points)):                                                  # Now the second part of the matrix with Vy
#        Sigma_inverse_squared[line+nb_points,line] = Sigma_inverse[line,1,0]            # Correlation of Vy with Vx
#        Sigma_inverse_squared[line+nb_points,line+nb_points] =  Sigma_inverse[line,1,1] # Correlation of Vy with Vy
#    K = Sigma_inverse_squared @ Hpiv_Topos   # We define K as the sigma inversed matrix times the Hpiv_Topos matrix
    
                

    
    
    # Calculating necessary matrices
    K= np.zeros((int(nb_dim),int(nb_modes+1),int(nb_points)))
    Sigma_inverse = np.transpose(Sigma_inverse,(1,2,0))
    Hpiv_Topos = np.reshape(Hpiv_Topos,(int(nb_points),int(nb_dim),int(nb_modes+1)),order='F') 
    Hpiv_Topos = np.transpose(Hpiv_Topos,(1,2,0))
    for line in range(int(nb_points)):                                                  # To all spatial samples we create the first part of the matrix that contains the correlation of Vx
        K[:,:,line] = Sigma_inverse[:,:,line] @ Hpiv_Topos[:,:,line]
    K = np.transpose(K,(2,0,1)) # ((nb_points),(nb_dim),(nb_modes+1),)
    K = np.reshape(K,(int(nb_points*nb_dim),int(nb_modes+1)),order='F') 
    Hpiv_Topos = np.transpose(Hpiv_Topos,(2,0,1)) # ((nb_points),(nb_dim),(nb_modes+1),)
    Hpiv_Topos = np.reshape(Hpiv_Topos,(int(nb_points*nb_dim),int(nb_modes+1)),order='F') 
    Sigma_inverse = np.transpose(Sigma_inverse,(2,0,1))  # ((nb_points),(nb_dim),(nb_dim),)
    
#    K = Sigma_inverse_squared @ Hpiv_Topos   # We define K as the sigma inversed matrix times the Hpiv_Topos matrix
    
    Hpiv_Topos_K = Hpiv_Topos.T @ K          # The Hpiv_Topos times K is necessary too
    
    
    print('Likelihood matrices computed')
    
#%%
    
    
    # LOAD dns simulation 
#    if type_data == 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated':
#        folder_DNS = 'folder_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks'
#        for number_of_file in range(81,81+number_of_files):
##        number_of_file = 81 # at least 81 because the others are used to train and max is 100
#        
#            file = 'file_DNS300_inc3d_3D_2017_04_02_'+ str(number_of_file)
#            Path_file_fluid_flow_DNS = Path(folder_data).parents[1].joinpath('data').joinpath(folder_DNS).joinpath(file)
#            data_DNS = hdf5storage.loadmat(str(Path_file_fluid_flow_DNS))
#            
#            if number_of_file == 81:
#                dt = data_DNS['dt'][0,0]
#                champ = data_DNS['U']
#                del data_DNS
#            else:
#                champ = np.concatenate((champ,data_DNS['U']),axis=3)
#                del data_DNS
#            
#            print('File number '+str(number_of_file)+' loaded')
        
#    elif type_data == 'DNS100_inc3d_2D_2018_11_16_blocks_truncated':
#        folder_DNS = 'folder_DNS100_inc3d_2D_2018_11_16_blocks'
#        number_of_file = 18 # at least 16 because the others are used to train and max is 20
#        file = 'file_DNS100_inc3d_2018_11_16_run_'+ str(number_of_file)
    
#    Path_file_fluid_flow_DNS = Path(folder_data).parents[1].joinpath('data').joinpath(folder_DNS).joinpath(file)
#    data_DNS = hdf5storage.loadmat(str(Path_file_fluid_flow_DNS))
#    champ = data_DNS['U']
#    dt = data_DNS['dt'][0,0]
#    param['dt'] = dt/n_simu # dt is coming from DNS before subsampling of topos 
#    param['N_test'] = n_simu*(champ.shape[3]-1)
#    
    ######################################################################
    
    
#    particles_estimate = np.zeros((1,bt_MCMC.shape[1]))
#    n_samples = int(5*1/param['dt'])
#    period_in_samples = int(5*1/param['dt'])
#    period = False
#    weigths_time_past = np.ones((bt_MCMC.shape[2]))/bt_MCMC.shape[2]
#    n,nb_pc1 = bt_MCMC.shape[1:]
#    shape_noise = (1,(n+1)*n,nb_pc1)
#    noises = np.zeros(shape=shape_noise)
   
#    observations_to_save = bt_tronc + beta_2*lambda_values[np.newaxis,...]*np.random.normal(0,1,size=(1,bt_tronc.shape[1]))
    
#    if assimilate_DNS == False:
#        factor = 3
#    else: 
#        factor = 1
    ###################################--calculate important matrices to assimilation #####################
    
#    reconstruc_flow = calculate_residue_and_return_reconstructed_flow(topos_l,champ,bt_tot)
#    tensor_var = load_cov_tensor(Path(folder_data).parents[1].joinpath('data').joinpath(Path(param['name_file_diffusion_mode']).name))
    
    
    
    
    
#    variance_uxx = tensor_var[:,0,0]
#    variance_uxy = tensor_var[:,0,1]
#    variance_uxz = tensor_var[:,0,2]
#    
#    variance_uyx = tensor_var[:,1,0]
#    variance_uyy = tensor_var[:,1,1]
#    variance_uyz = tensor_var[:,1,2]
#    
#    variance_uzx = tensor_var[:,2,0]
#    variance_uzy = tensor_var[:,2,1]
#    variance_uzx = tensor_var[:,2,2]
    
    
    
#    sigma = np.concatenate((variance_ux,variance_uy,variance_uz))[...,np.newaxis]
    
#    if mask == True:
#        for i in range(len(param['MX'][0,:])):
#            param['MX'][0,i] = math.ceil(param['MX'][0,i]/subsampling_grid)
#            
#            
#            
##        matrix_H = matrix_H.reshape((champ.shape[0],champ.shape[1],champ.shape[2],champ.shape[4],matrix_H.shape[1]),order='F')
##        sigma = sigma.reshape((champ.shape[0],champ.shape[1],champ.shape[2],champ.shape[4],sigma.shape[1]),order='F')
#        topos_l = topos_l.reshape((champ.shape[0],champ.shape[1],champ.shape[2],champ.shape[4],topos_l.shape[1]),order='F')
#        
#        
##        matrix_H = matrix_H[::subsampling_grid,::subsampling_grid,::subsampling_grid,:,:]
##        sigma = sigma[::subsampling_grid,::subsampling_grid,::subsampling_grid,:,:]
#        champ = champ[::subsampling_grid,::subsampling_grid,::subsampling_grid,:]
#        topos_l = topos_l[::subsampling_grid,::subsampling_grid,::subsampling_grid,:]
#        
##        matrix_H = matrix_H.reshape((matrix_H.shape[0]*matrix_H.shape[1]*matrix_H.shape[2]*matrix_H.shape[3],matrix_H.shape[4]),order='F')
##        sigma = sigma.reshape((sigma.shape[0]*sigma.shape[1]*sigma.shape[2]*sigma.shape[3],sigma.shape[4]),order='F')
#        topos_l = topos_l.reshape((topos_l.shape[0]*topos_l.shape[1]*topos_l.shape[2]*topos_l.shape[3],topos_l.shape[4]),order='F')
  
#    np.save('grid.npy',param['grid'][0])
#    if dim==2:
#        grid = np.concatenate((topos.shape[:2],np.array([1])))
#    elif dim==3:
#        grid = topos.shape[:3]
        
    
    
    
    
#    teste_matrix = np.ravel(matrix_H,order='F')
#    error = np.subtract(teste_topos, teste_matrix)
#    pos = np.where((error!=0))[0]
    
#    topos_l_res = topos_l.reshape((grid[0],grid[1],grid[2],dim,nb_modes+1),order='F')

#    topos_l_res = topos_l_res[number:-number,number:-number,...]
#    topos_l_res_ens = topos_l_res.reshape((int(topos_l_res.shape[0]*topos_l_res.shape[1]*topos_l_res.shape[2]*topos_l_res.shape[3]),int(topos_l_res.shape[4])),order='F')
#    topos_l_res_ux = topos_l_res[number:-number,:,slice_z:slice_z+1,0,:]
#    topos_l_res_uy = topos_l_res[:,number:-number,slice_z:slice_z+1,1,:]
#    topos_l_res_uz = topos_l_res[:,:,slice_z:slice_z+1,2,:]
#    
#    topos_l_res_ux = topos_l_res_ux.reshape((topos_l_res_ux.shape[0]*topos_l_res_ux.shape[1]*topos_l_res_ux.shape[2],topos_l_res_ux.shape[3]),order='F')
#    topos_l_res_uy = topos_l_res_uy.reshape((topos_l_res_uy.shape[0]*topos_l_res_uy.shape[1]*topos_l_res_uy.shape[2],topos_l_res_uy.shape[3]),order='F')
#    topos_l_res_uz = topos_l_res_uz.reshape((topos_l_res_uz.shape[0]*topos_l_res_uz.shape[1]*topos_l_res_uz.shape[2],topos_l_res_uz.shape[3]),order='F')
#    
#    topos_l_res_ens = np.concatenate((topos_l_res_ux,topos_l_res_uy,topos_l_res_uz))
#    
#    error_matrix = np.subtract(matrix_H,topos_l_res_ens)
#    pos = np.where((error_matrix!=0))[0]
#    error = np.sum(np.power(error_matrix,2),axis=0)
    
    
    
    
#    dt_optimal = param['dt']*param_ref['n_simu']
#    Sigma_inversed_diagonal = calculate_noise_covariance_tensor(tensor_var,L,param['dX'],param['MX'],std_space,subsampling_grid,dim,dt_optimal,slicing,slice_z)

    
    
#    if Sigma_inversed_diagonal.shape[1] == 1:
#        K = (np.multiply(Sigma_inversed_diagonal,matrix_H)).T
#    else:
#        K = (np.matmul(Sigma_inversed_diagonal,matrix_H)).T
#
#    
##    u, s, vh = np.linalg.svd(K)
#    
#    
#    Hr_K = np.matmul(matrix_H.T,K.T)
    
#   ##########################  
    
#%%                    LOAD   Data PIV 
    
    
    if assimilate == 'real_data':
    
        file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('wake_Re'+str(Re)).joinpath('B'+str(1).zfill(4)+'.dat')   # The path to load PIV data
        data = open(str(file))                                                                                                     # Open the PIV data  
        datContent = [i.strip().split() for i in data.readlines()]                                                                 # Reading the data 
        data = datContent[4:]                                                                                                      # Getting the data PIV
        
        nb_lines = len(data)                           # lines amount
        nb_collums = len(data[0][2:4])                 # Commumns amount 
        matrix = np.zeros(shape=(nb_lines,nb_collums)) # Creating matrix to stock data 
        
        '''
        We will select the first data in 0.080833 and after we will load the files taking in account the factor of subsampling and the amount of files. 
        '''
        for i,line in enumerate(data):  # Select the data in the first PIV file 
            for j,number in enumerate(line[2:4]):
                matrix[i,j] = number
        
        matrix_data_PIV_all_data = matrix[Sigma_inverse_data['mask'][:,0],:].copy()[np.newaxis,...] # Select the PIV data in the first mask calculated (The mask inside PIV and DNS)    
        print('Loading PIV data: '+str(number_of_PIV_files)+' files...')
        
        if number_of_PIV_files>1:
            for nb_file in range(1,(number_of_PIV_files+1),factor_of_PIV_time_subsampling)[1:]:                                                    # Loading the other files as defined in the start of this function
                print(nb_file)
                file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('wake_Re'+str(Re)).joinpath('B'+str(nb_file).zfill(4)+'.dat') # Path to file
                data = open(str(file))                                                                                                         # Open the file       
                datContent = [i.strip().split() for i in data.readlines()]                                                                     # Read the data                                                                                                                                                                              
                data = datContent[4:]                                                                                                          # Getting the data PIV 
                
                
                matrix = np.zeros(shape=(nb_lines,nb_collums)) # Define the matrix to stock the data  
                for i,line in enumerate(data):                 # Decode the information and save it as a matrix
                    for j,number in enumerate(line[2:4]):
                        matrix[i,j] = number
                
            
                matrix_data_PIV_all_data = np.concatenate((matrix_data_PIV_all_data,matrix[Sigma_inverse_data['mask'][:,0],:].copy()[np.newaxis,...]),axis=0)  # Save the matrix inside the matrix of all the PIV data
        
        
        # Normalizing  measured data
        matrix_data_PIV_all_data = matrix_data_PIV_all_data/u_inf_measured  # Normalizing the PIV data to compare with DNS 
    #    np.save('Data_piv.npy',matrix_data_PIV_all_data)                   # If necessary to save(This will be saved as numpy array in this folder.)
        
        '''
        We need to apply the same observation mask on the observed data. Because the new mask is defined to control where we will observe inside the PIV window
        '''
        matrix_data_PIV_all_data = matrix_data_PIV_all_data[:,Mask_final_bool[:matrix_data_PIV_all_data.shape[1]],:].copy()  
        
        
    elif assimilate == 'fake_real_data':
        '''
        The case of assimilate fake_real_data is choosen when it should be assimilated the DNS smoothed and with noise. Hence, 
        a fake PIV(real data).
        '''
        i=1
        file = (Path(__file__).parents[3]).joinpath('data_PIV')\
        .joinpath('wake_Re'+str(Re)+'_fake')\
        .joinpath('strat'+str(nb_file_learning_basis+i)+'_U_temp_PIV')   # The path to load PIV data
        data = hdf5storage.loadmat(str(file)) 
        vector_of_assimilation_time = np.array(data['interval_time_local'][0,:])
        
        dt_PIV = np.array(data['dt'][0,:])
        index_final = SECONDS_OF_SIMU/dt_PIV
        vector_flow = data['U'].copy()
        
        nb_snapshots_each_file = vector_flow.shape[1]
        time_per_file = (nb_snapshots_each_file)*dt_PIV
        if SECONDS_OF_SIMU%time_per_file == 0:
            number_of_FAKE_PIV_files = int(SECONDS_OF_SIMU/time_per_file)
#            number_of_FAKE_PIV_files = intSECONDS_OF_SIMU/time_per_file
        else:
            number_of_FAKE_PIV_files = int(SECONDS_OF_SIMU/time_per_file) + 1
        
        print('Loading Fake PIV data:'+str(number_of_FAKE_PIV_files)+' files ...')
        if number_of_FAKE_PIV_files>1:
            for i in range(1,number_of_FAKE_PIV_files+1,1)[1:]:
                file = (Path(__file__).parents[3]).joinpath('data_PIV')\
                .joinpath('wake_Re'+str(Re)+'_fake')\
                .joinpath('strat'+str(nb_file_learning_basis+i)+'_U_temp_PIV')   # The path to load PIV data
                data = hdf5storage.loadmat(str(file)) 
                vector_flow = np.concatenate((vector_flow,data['U']),axis=1)
                vector_of_assimilation_time = np.concatenate((vector_of_assimilation_time,np.array(data['interval_time_local'][0,:])))
        
        vector_flow = np.transpose(vector_flow,(1,0,2))
        vector_flow = vector_flow[:int(index_final):factor_of_PIV_time_subsampling,:,:]
        vector_flow = vector_flow[:,Mask_final_bool[:vector_flow.shape[1]],:]
        matrix_data_PIV_all_data = vector_flow.copy()
     
        vector_of_assimilation_time = vector_of_assimilation_time[:int(index_final):factor_of_PIV_time_subsampling]
        
        # Plots for debug
        if plot_debug:
            mean_vector_flow = np.mean(vector_flow, axis=0)
            mean_vector_flow = np.reshape(mean_vector_flow,(len(coordinates_x_PIV),len(coordinates_y_PIV),2),order='F') 
            mean_vector_flow.shape
            fig=plt.figure()
            imgplot = plt.imshow(np.transpose(mean_vector_flow[:,:,0]),interpolation='none', extent=[coordinates_x_PIV[0],coordinates_x_PIV[-1],coordinates_y_PIV[0],coordinates_y_PIV[-1]])
            plt.pause(1) 
            fig.colorbar(imgplot)
            plt.pause(1) 
        
            differ = mean_vector_flow[:,:,1] - matrix_H_plot[:,:,1,-1]
            fig=plt.figure(3)
            imgplot = plt.imshow(np.transpose(differ),interpolation='none', extent=[coordinates_x_PIV[0],coordinates_x_PIV[-1],coordinates_y_PIV[0],coordinates_y_PIV[-1]])
            plt.pause(1) 
            fig.colorbar(imgplot)
            plt.pause(1) 
        
#        if len(vector_of_assimilation_time) > 0:
#            if vector_of_assimilation_time[0] == 0:
#                matrix_data_PIV_all_data[1:,:,:]
##                vector_of_assimilation_time = vector_of_assimilation_time[1:]
#        else:
#            vector_of_assimilation_time = np.full((1, 1), np.inf)
            
#        if len(vector_of_assimilation_time) == 0:
#            vector_of_assimilation_time = np.full((1, 1), np.inf)
        
    else:
        print('Error: Data to assimilate is not known.')
        sys.exit()
#    vector_of_assimilation_time = vector_of_assimilation_time[2:]
        
    
    if len(vector_of_assimilation_time) > 0:
        if vector_of_assimilation_time[0] == 0:
            matrix_data_PIV_all_data=matrix_data_PIV_all_data[1:,:,:]
            vector_of_assimilation_time = vector_of_assimilation_time[1:]
    else:
        vector_of_assimilation_time = np.full((1, 1), np.inf)
        
    #%% Pre-processing for plotting Q cirterion
    if plot_Q_crit:
        file = 'tensor_mode_' + type_data + '_'+str(nb_modes)+'_modes'
        name_file_data = Path(__file__).parents[3].joinpath('data').joinpath(file)
        mat = hdf5storage.loadmat(str(name_file_data))
        Omega_phi_m_U = mat['Omega_phi_m_U']
        S_phi_m_U = mat['S_phi_m_U']
    
    #%% Begin propagation and assimilation
    pchol_cov_noises = beta_3*pchol_cov_noises                           # Cholesky de la matrix de covariance                          
    if EV:
        ILC_EV['pchol_cov_noises'] = beta_3*ILC_EV['pchol_cov_noises']
    original_dt_simu = param['dt']                                       # Time simulation step  
    assimilate_PIV = False                                               # Flag to control assimilation moments
    nb_assim = 0                                                         # Flag to count the assimilation steps 
    next_time_of_assimilation = vector_of_assimilation_time[nb_assim]    # Control de next moment that a obs will be available
    index_of_filtering = []                                              # Control de index of assimilation
    time = [0]                                                           # The time of assimilation
    index_pf = [0]                                                       # Flag to control de noise in the past until now to mutation steps in Metropolis-Hastings
#    time_bt_tot = time_bt_tot + next_time_of_assimilation
    
    # Defining figure to plot if real data is True 
    if plt_real_time==True:
        plt.ion()
        fig = plt.figure(0,figsize =[fig_width,fig_height])
        plt.rcParams['axes.grid'] = True
        
        
        ax_1 = fig.add_subplot(2,2,1)
        ax_1.set_ylim([-10, 10])
        
        ax_2 = fig.add_subplot(2,2,2)
        ax_2.set_ylim([-10, 10])
        
        ax_3 = fig.add_subplot(2,4,5)
        ax_3.set_ylim([-10, 10])
        
        ax_4 = fig.add_subplot(2,4,6)
        ax_4.set_ylim([-10, 10])
        
        ax_5 = fig.add_subplot(2,4,7)
        ax_5.set_ylim([-10, 10])
        
        ax_6 = fig.add_subplot(2,4,8)
        ax_6.set_ylim([-10, 10])
        
        quantiles_now = np.quantile(bt_MCMC[-1,:,:],q=[0.025,0.975],axis=1)
        particles_mean_now = np.mean(bt_MCMC[-1,:,:],axis=1)
        if EV:
            quantiles_now_EV = np.quantile(bt_forecast_EV[-1,:,:],q=[0.025,0.975],axis=1)
            particles_mean_now_EV = np.mean(bt_forecast_EV[-1,:,:],axis=1)
        
        line11, = ax_1.plot(time[-1], particles_mean_now[0], 'b-',label = 'Red LUM particles mean')
        line12  = ax_1.fill_between([0], quantiles_now[0:1,0],quantiles_now[1:2,0], color='gray')
        if EV:
            line11EV, = ax_1.plot(time[-1], particles_mean_now_EV[0], '-', \
                                    color=color_mean_EV,label = 'EV particles mean')
            line12EV  = ax_1.fill_between([0], quantiles_now_EV[0:1,0],quantiles_now[1:2,0], color=color_quantile_EV)
        line13,  = ax_1.plot([0],[pos_Mes*1],'r.',label = 'Assimilate True')
        
        
        
        line21, = ax_2.plot(time[-1], particles_mean_now[1], 'b-',label = 'Red LUM particles mean')
        if heavy_real_time_plot:
            line22 = ax_2.fill_between([0], quantiles_now[0:1,1],quantiles_now[1:2,1], color='gray')
        if EV:
            line21EV, = ax_2.plot(time[-1], particles_mean_now_EV[1], '-', \
                                    color=color_mean_EV,label = 'EV particles mean')
            if heavy_real_time_plot:
                line22EV  = ax_2.fill_between([0], quantiles_now_EV[0:1,1],quantiles_now[1:2,1], color=color_quantile_EV)
        line23,  = ax_2.plot([0],[pos_Mes*1],'r.',label = 'Assimilate True')
       
        
        
        if bt_tot.shape[1]>2:
            line31, = ax_3.plot(time[-1], particles_mean_now[2], 'b-',label = 'Red LUM particles mean')
            if heavy_real_time_plot:
                line32 = ax_3.fill_between([0], quantiles_now[0:1,2],quantiles_now[1:2,2], color='gray')
            if EV:
                line31EV, = ax_3.plot(time[-1], particles_mean_now_EV[2], '-', \
                                        color=color_mean_EV,label = 'EV particles mean')
                if heavy_real_time_plot:
                    line32EV  = ax_3.fill_between([0], quantiles_now_EV[0:1,2],quantiles_now[1:2,2], color=color_quantile_EV)
            line33,  = ax_3.plot([0],[pos_Mes*1],'r.',label = 'Assimilate True')
        
        if bt_tot.shape[1]>3:
            line41, = ax_4.plot(time[-1], particles_mean_now[3], 'b-',label = 'Red LUM particles mean')
            if heavy_real_time_plot:
                line42 = ax_4.fill_between([0], quantiles_now[0:1,3],quantiles_now[1:2,3], color='gray')
            if EV:
                line41EV, = ax_4.plot(time[-1], particles_mean_now_EV[3], '-', \
                                        color=color_mean_EV,label = 'EV particles mean')
                if heavy_real_time_plot:
                    line42EV  = ax_4.fill_between([0], quantiles_now_EV[0:1,3],quantiles_now[1:2,3], color=color_quantile_EV)
            line43,  = ax_4.plot([0],[pos_Mes*1],'r.',label = 'Assimilate True')

        if bt_tot.shape[1]>4:
            line51, = ax_5.plot(time[-1], particles_mean_now[4], 'b-',label = 'Red LUM particles mean')
            if heavy_real_time_plot:
                line52 = ax_5.fill_between([0], quantiles_now[0:1,4],quantiles_now[1:2,4], color='gray')
            if EV:
                line51EV, = ax_5.plot(time[-1], particles_mean_now_EV[4], '-', \
                                        color=color_mean_EV,label = 'EV particles mean')
                if heavy_real_time_plot:
                    line52EV  = ax_5.fill_between([0], quantiles_now_EV[0:1,4],quantiles_now[1:2,4], color=color_quantile_EV)
            line53,  = ax_5.plot([0],[pos_Mes*1],'r.',label = 'Assimilate True')

        if bt_tot.shape[1]>5:
            line61, = ax_6.plot(time[-1], particles_mean_now[5], 'b-',label = 'Red LUM particles mean')
            if heavy_real_time_plot:
                line62 = ax_6.fill_between([0], quantiles_now[0:1,5],quantiles_now[1:2,5], color='gray')
            if EV:
                line61EV, = ax_6.plot(time[-1], particles_mean_now_EV[5], '-', \
                                        color=color_mean_EV,label = 'EV particles mean')
                if heavy_real_time_plot:
                    line6EV2  = ax_6.fill_between([0], quantiles_now_EV[0:1,5],quantiles_now[1:2,5], color=color_quantile_EV)
            line63,  = ax_6.plot([0],[pos_Mes*1],'r.',label = 'Assimilate True')
        
        
        if plot_ref==True:
            line14, =  ax_1.plot(time_bt_tot[-1],bt_tot[0,0],'k--',label = 'True state')
            line24, =  ax_2.plot(time_bt_tot[-1],bt_tot[0,1],'k--',label = 'True state')
            if bt_tot.shape[1]>2:
                line34, =  ax_3.plot(time_bt_tot[-1],bt_tot[0,2],'k--',label = 'True state')
            if bt_tot.shape[1]>3:
                line44, =  ax_4.plot(time_bt_tot[-1],bt_tot[0,3],'k--',label = 'True state')
            if bt_tot.shape[1]>4:
                line54, =  ax_5.plot(time_bt_tot[-1],bt_tot[0,4],'k--',label = 'True state')
            if bt_tot.shape[1]>5:
                line64, =  ax_6.plot(time_bt_tot[-1],bt_tot[0,5],'k--',label = 'True state')
        
        
        ax_1.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(1)+'$'+' amplitude')
        ax_1.legend()
        ax_2.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(2)+'$'+' amplitude')
        ax_2.legend()
        if bt_tot.shape[1]>2:
            ax_3.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(3)+'$'+' amplitude')
            ax_3.legend()
        if bt_tot.shape[1]>3:
            ax_4.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(4)+'$'+' amplitude')
            ax_4.legend()
        if bt_tot.shape[1]>4:
            ax_5.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(5)+'$'+' amplitude')
            ax_5.legend()
        if bt_tot.shape[1]>5:
            ax_6.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(6)+'$'+' amplitude')
            ax_6.legend()
    

    
    N_tot_max = int(SECONDS_OF_SIMU/param['dt'])+1
    N_tot = param['N_tot']*n_simu
    if N_tot > N_tot_max:
        N_tot = N_tot_max
    param['N_tot'] = N_tot
    param['N_test'] = param['N_tot'] - 1
    time_exe = 0
    if EV:
        time_exe_EV = 0
    n_frame_plots = int(plot_period/param['dt'])
    
                   
    ################################ Start temporal integration ###################################
    for index in range(param['N_test']): # Set the number of integration steps

        ##### Model integration of all particles
#        val0,val1,val2,noises_centered = evol_forward_bt_MCMC(ILC_a_cst['modal_dt']['I'],\
#                                                        ILC_a_cst['modal_dt']['L'],\
#                                                        ILC_a_cst['modal_dt']['C'],\
#                                                        pchol_cov_noises,param['dt'],\
#                                                        bt_MCMC[-1,:,:],float('Nan'),\
#                                                        float('Nan'),mutation=False,noise_past=0,pho=0)
        start = t_exe.time()
        val0,val1,val2,noises_centered = evol_forward_bt_MCMC(ILC_a_cst['I'],\
                                                        ILC_a_cst['L'],\
                                                        ILC_a_cst['C'],\
                                                        pchol_cov_noises,param['dt'],\
                                                        bt_MCMC[-1,:,:],float('Nan'),\
                                                        float('Nan'),mutation=False,noise_past=0,pho=0)
        end = t_exe.time()
        time_exe = time_exe + end - start
        if EV:
            start = t_exe.time()
            val0_EV,val1_EV,val2_EV,noises_centered_EV = evol_forward_bt_MCMC(ILC_EV['I'],\
                                                            ILC_EV['L'],\
                                                            ILC_EV['C'],\
                                                            ILC_EV['pchol_cov_noises'],param['dt'],\
                                                            bt_forecast_EV[-1,:,:],float('Nan'),\
                                                            float('Nan'),mutation=False,noise_past=0,pho=0)
            end = t_exe.time()
            time_exe_EV = time_exe_EV + end - start
        
        time.append(param['dt']+time[-1])
        #########################################----------------------#############################################
        #########################################--PARTICLE FILTERING--#############################################

#        if (index+1)%(int(period_in_samples*beta_4))== 0:
#            period = True
#            print('Index of activating filtering: '+str(index))
        
#        if ((index+1))%(int(factor*n_simu*beta_4)) == 0:
#            period = True
#            print('-----------------------------STARTING PARTICLE FILTERING---------------------------')
#            print('Index of activating filtering: '+str(index))
#        period = True
        
    
        
        
        
        if (assimilate_PIV==True):                      # The Flag assimilate_PIV control the moments that we can assimilate data
            index_of_filtering.append(index)            # Stock the assimilation index
            print('Index of filtering: '+str(index))    
            index_pf.append(index+1)                    # Stock the assimilation index to control noise and past particles

#            obs = bt_tot[ int((index+1)/n_simu),:][...,np.newaxis]
#            index_assimilation = (index+1)/(n_simu)
#            if assimilate_DNS == False:
#                obs = reconstruc_flow[:,:,:,:,int(index_assimilation)]
#                
#                
#            else:
#                obs = champ[:,:,:,int(index_assimilation),:]
            
            # Define the obs and reshape it
            obs = np.reshape(matrix_data_PIV_all_data[nb_assim,:,:],(matrix_data_PIV_all_data[nb_assim,:,:].shape[0]*matrix_data_PIV_all_data[nb_assim,:,:].shape[1]),order='F')[...,np.newaxis]
            
            particles = val0[0,:,:]                     # Define the particles now
            particles_past = bt_MCMC[index_pf[-2],...]  # Define the particles after the last filter step
            if EV:
                particles_EV = val0_EV[0,:,:]                     # Define the particles now
                particles_past_EV = bt_forecast_EV[index_pf[-2],...]  # Define the particles after the last filter step
            delta_t = index_pf[-1] - index_pf[-2]       # Define the delta t as the number of integrations(IMPORTANT: In the case of real time assimilation the dt is variable.....)
            
            # Call particle filter 
            start = t_exe.time()
            particles = particle_filter(ILC_a_cst,obs,K,Hpiv_Topos_K,particles,N_threshold,\
                                        np.concatenate((noises,noises_centered[np.newaxis,...]),axis=0)[index_pf[-2]:index_pf[-1],...],\
                                        particles_past,nb_mutation_steps,original_dt_simu,param['dt'],pho,delta_t,pchol_cov_noises,time[-1]) 
            end = t_exe.time()
            time_exe = time_exe + end - start
            if EV:
                start = t_exe.time()
                particles_EV = particle_filter(ILC_EV,obs,K,Hpiv_Topos_K,particles_EV,N_threshold,\
                                        np.concatenate((noises_EV,noises_centered_EV[np.newaxis,...]),axis=0)[index_pf[-2]:index_pf[-1],...],\
                                        particles_past_EV,nb_mutation_steps,original_dt_simu,param['dt'],pho,delta_t,ILC_EV['pchol_cov_noises'],time[-1]) 
                end = t_exe.time()
                time_exe_EV = time_exe_EV + end - start
                                        
                                        
            
 
            val0 = particles[np.newaxis,...]                                         # Define the particles 
            if EV:
                val0_EV = particles_EV[np.newaxis,...]                                         # Define the particles 
            param['dt'] = original_dt_simu                                           # Set the integration step as the original one 
            if (nb_assim ) == len(vector_of_assimilation_time)-1:                    # Set the next time of assimilation in the inf if there is no more data in the vecor
                next_time_of_assimilation = np.inf
            else:                                                                    # If there is data
                nb_assim += 1                                                        # Increments the Flag that controls the assimilation steps
                next_time_of_assimilation = vector_of_assimilation_time[nb_assim]    # Set the next time of assimilation
                
                
            assimilate_PIV = False      # Set the control Flag to False                                                                           

        ############################################################################################################
        #############################################################################################################
        if index==0:                                                                    # If the first time step integration
            noises = noises_centered[np.newaxis,...]                                    # Set the noises
        elif (next_time_of_assimilation != np.inf):                                     # If the next time of integration is not at inf, it saves the noise now
            noises = np.concatenate((noises,noises_centered[np.newaxis,...]),axis=0) 
        if EV:
            if index==0:                                                                    # If the first time step integration
                noises_EV = noises_centered_EV[np.newaxis,...]                                    # Set the noises
            elif (next_time_of_assimilation != np.inf):                                     # If the next time of integration is not at inf, it saves the noise now
                noises_EV = np.concatenate((noises_EV,noises_centered_EV[np.newaxis,...]),axis=0)
            
        bt_MCMC = np.concatenate((bt_MCMC,val0),axis=0)    # Concatenate the particles in this time step with the particles before
#        bt_fv   = np.concatenate((bt_fv,val1),axis=0)
#        bt_m    = np.concatenate((bt_m,val2),axis=0)
        iii_realization = np.any(np.logical_or(np.isnan(bt_MCMC[index+1,:,:]),np.isinf(bt_MCMC[index+1,:,:])),axis = 0)[...,np.newaxis]  # Control if any realization has explosed
        
        if EV:
            bt_forecast_EV = np.concatenate((bt_forecast_EV,val0_EV),axis=0)    # Concatenate the particles in this time step with the particles before
            iii_realization_EV = np.any(np.logical_or(np.isnan(bt_forecast_EV[index+1,:,:]),np.isinf(bt_forecast_EV[index+1,:,:])),axis = 0)[...,np.newaxis]  # Control if any realization has explosed


        if (time[-1]+param['dt'])>=(next_time_of_assimilation):  # If the next time integration will end after the time of assimilation, hence we need to change the time step 'dt' to end exactly in the same time of the observation
            param['dt'] = next_time_of_assimilation - time[-1]   # Therefore, the next time integration step will be the difference between the future and the present.
            assimilate_PIV = True                                # Set the Flag True
            
        
            ############################ Solve possible explosions in the integration
        if np.any(iii_realization):
            if np.all(iii_realization):
                print('WARNING: All realization of the simulation have blown up.')
                
                if index < param['N_test']:
                    val_nan = np.full([int(param['N_test']-index), param['nb_modes'], param['N_particules']], np.nan)
                    bt_MCMC = np.concatenate((bt_MCMC,val_nan),axis=0)    
#                    bt_fv   = np.concatenate((bt_fv,val_nan),axis=0)
#                    bt_m    = np.concatenate((bt_m,val_nan),axis=0)
                
                break 
            
            
        
            nb_blown_up = np.sum(iii_realization)
            print('WARNING: '+ str(nb_blown_up)+' realizations have blown up and will be replaced.')
            good_indexes = np.where((np.logical_not(iii_realization) == True))[0]
            bt_MCMC_good = bt_MCMC[-1,:, good_indexes].T
#            bt_fv_good = bt_fv[-1,:, good_indexes].T
#            bt_m_good = bt_m[-1,:, good_indexes].T
            
            
            bt_MCMC_good = bt_MCMC_good[np.newaxis,...]
#            bt_fv_good = bt_fv_good[np.newaxis,...]
#            bt_m_good = bt_m_good[np.newaxis,...]
                
            
            rand_index =  np.random.randint(0, param['N_particules'] - nb_blown_up, size=(nb_blown_up))
            
#            if rand_index.shape == (1,1):
#                rand_index = rand_index[0,0]
                
                
            bad_indexes = np.where((iii_realization == True))[0]
            bt_MCMC[-1,:, bad_indexes] = bt_MCMC_good[0,:, rand_index]  
#            bt_fv[-1,:, bad_indexes] = bt_fv_good[0,:, rand_index]
#            bt_m[-1,:, bad_indexes] = bt_m_good[0,:, rand_index]
    
            del bt_MCMC_good 
            del rand_index 
            del nb_blown_up 
            del iii_realization     
        
        if EV:
                        ############################ Solve possible explosions in the integration
            if np.any(iii_realization_EV):
                if np.all(iii_realization_EV):
                    print('WARNING: All realization of the simulation have blown up.')
                    if index < param['N_test']:
                        val_nan_EV = np.full([int(param['N_test']-index), param['nb_modes'], param['N_particules']], np.nan)
                        bt_forecast_EV = np.concatenate((bt_forecast_EV,val_nan),axis=0)    
                    break 

                nb_blown_up_EV = np.sum(iii_realization_EV)
                print('WARNING: '+ str(nb_blown_up_EV)+' realizations have blown up and will be replaced.')
                good_indexes_EV = np.where((np.logical_not(iii_realization_EV) == True))[0]
                bt_forecast_EV_good = bt_forecast_EV[-1,:, good_indexes_EV].T
                
                
                bt_forecast_EV_good = bt_forecast_EV_good[np.newaxis,...]
#                bt_fv_good = bt_fv_good[np.newaxis,...]
#                bt_m_good = bt_m_good[np.newaxis,...]
                    
                
                rand_index_EV =  np.random.randint(0, param['N_particules'] - nb_blown_up_EV, size=(nb_blown_up_EV))
                
    #            if rand_index.shape == (1,1):
    #                rand_index = rand_index[0,0]
                    
                    
                bad_indexes_EV = np.where((iii_realization_EV == True))[0]
                bt_forecast_EV_good[-1,:, bad_indexes_EV] = bt_forecast_EV_good[0,:, rand_index_EV]  
#                bt_fv[-1,:, bad_indexes] = bt_fv_good[0,:, rand_index]
#                bt_m[-1,:, bad_indexes] = bt_m_good[0,:, rand_index]
        
                del bt_forecast_EV_good 
                del rand_index_EV 
                del nb_blown_up_EV 
                del iii_realization_EV   
    
        
        #%% Plots
        
            
        ######################### Testing real time plot #######################
        
        if (index%n_frame_plots)==0 and ( (plot_Q_crit==True) or (plt_real_time==True)):   # Plot at each 20 time steps 
#        if (index%n_frame_plots)==0 and (index!=0) and (plt_real_time==True):   # Plot at each 20 time steps 
#        if (index%20)==0 and (index!=0) and (plt_real_time==True):   # Plot at each 20 time steps 
            particles_mean = np.mean(bt_MCMC[:,:,:],axis=2)
            lim = np.where((time_bt_tot<=time[-1]))[0][-1] + 1
            
            if EV:
                particles_mean_EV = np.mean(bt_forecast_EV[:,:,:],axis=2)
                if (plt_real_time==True):
                    quantiles_EV = np.quantile(bt_forecast_EV[:,:,:],q=[0.025,0.975],axis=2)
                
                ########################## Plotting Q cirterion ########################
                if plot_Q_crit:
    #                particles_mean = mat['particles_mean']
                    particles_mean_ = np.hstack((particles_mean_EV,np.ones((particles_mean_EV.shape[0],1))))[index,:]
                    particles_mean_ = np.tile(particles_mean_,([Omega_phi_m_U.shape[0],Omega_phi_m_U.shape[2],Omega_phi_m_U.shape[3],1]))
                    particles_mean_ = np.transpose(particles_mean_,(0,3,1,2))
                    
                    Omega = np.multiply(particles_mean_,Omega_phi_m_U)
                    Omega = np.sum(Omega,axis=1)
                    Omega = np.sum(np.sum(np.power(Omega,2),axis=2),axis=1)
                    
                    S = np.multiply(particles_mean_,S_phi_m_U)
                    S = np.sum(S,axis=1)
                    S = np.sum(np.sum(np.power(S,2),axis=2),axis=1)
                    
                    Q = 0.5 * ( Omega - S )
                    del Omega
                    del S
                    
                    MX = param['MX'].astype(int)
                    Q = np.reshape(Q,(MX))
                    
                    time_CPU = time_exe_EV
                    time_phy = index*param['dt']
                    time_CPU_phy = np.array([time_CPU, time_phy])
                    
    #                file = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath('aurore')
                    name_file_data_temp = path_Q_crit_EV.joinpath(str(index)+'_temp.txt')
                    name_file_data = path_Q_crit_EV.joinpath(str(index)+'.txt')
                    
                    if os.path.exists(str(name_file_data)):
                        os.remove(str(name_file_data))
                    with open(str(name_file_data_temp), 'w') as f:
                        json.dump(time_CPU_phy.tolist(), f)
                        json.dump(Q.tolist(), f)
                    os.rename(r'' + str(name_file_data_temp),r'' + str(name_file_data))
                    del Q
                
            if plot_ref:                
                ########################## Plotting Q cirterion ########################
                if plot_Q_crit:
                    bt_val_interp = np.zeros((1,bt_tot.shape[1]))
                    for j in range(bt_tot.shape[1]):
                        tck = interpolate.splrep(time_bt_tot, bt_tot[:,j], s=0)
                        bt_val_interp[0,j] = interpolate.splev(time[-1], tck, der=0)
    #                particles_mean = mat['particles_mean']
                    particles_mean_ = np.hstack((bt_val_interp,np.ones((1,1))))
                    particles_mean_ = np.tile(particles_mean_,([Omega_phi_m_U.shape[0],Omega_phi_m_U.shape[2],Omega_phi_m_U.shape[3],1]))
                    particles_mean_ = np.transpose(particles_mean_,(0,3,1,2))
                    
                    Omega = np.multiply(particles_mean_,Omega_phi_m_U)
                    Omega = np.sum(Omega,axis=1)
                    Omega = np.sum(np.sum(np.power(Omega,2),axis=2),axis=1)
                    
                    S = np.multiply(particles_mean_,S_phi_m_U)
                    S = np.sum(S,axis=1)
                    S = np.sum(np.sum(np.power(S,2),axis=2),axis=1)
                    
                    Q = 0.5 * ( Omega - S )
                    del Omega
                    del S
                    
                    MX = param['MX'].astype(int)
                    Q = np.reshape(Q,(MX))
                    
                    time_CPU = 0
                    time_phy = index*param['dt']
                    time_CPU_phy = np.array([time_CPU, time_phy])
                    
    #                file = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath('aurore')
                    name_file_data_temp = path_Q_crit_ref.joinpath(str(index)+'_temp.txt')
                    name_file_data = path_Q_crit_ref.joinpath(str(index)+'.txt')
                    
                    if os.path.exists(str(name_file_data)):
                        os.remove(str(name_file_data))
                    with open(str(name_file_data_temp), 'w') as f:
                        json.dump(time_CPU_phy.tolist(), f)
                        json.dump(Q.tolist(), f)
                    os.rename(r'' + str(name_file_data_temp),r'' + str(name_file_data))
                    del Q
            
            ########################## Plotting Q cirterion ########################
            if plot_Q_crit:
#                particles_mean = mat['particles_mean']
                particles_mean_ = np.hstack((particles_mean,np.ones((particles_mean.shape[0],1))))[index,:]
                particles_mean_ = np.tile(particles_mean_,([Omega_phi_m_U.shape[0],Omega_phi_m_U.shape[2],Omega_phi_m_U.shape[3],1]))
                particles_mean_ = np.transpose(particles_mean_,(0,3,1,2))
                
                Omega = np.multiply(particles_mean_,Omega_phi_m_U)
                Omega = np.sum(Omega,axis=1)
                Omega = np.sum(np.sum(np.power(Omega,2),axis=2),axis=1)
                
                S = np.multiply(particles_mean_,S_phi_m_U)
                S = np.sum(S,axis=1)
                S = np.sum(np.sum(np.power(S,2),axis=2),axis=1)
                
                Q = 0.5 * ( Omega - S )
                del Omega
                del S
                
                MX = param['MX'].astype(int)
                Q = np.reshape(Q,(MX))
                
                time_CPU = time_exe
                time_phy = index*param['dt']
                time_CPU_phy = np.array([time_CPU, time_phy])
                
#                file = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath('aurore')
                name_file_data_temp = path_Q_crit.joinpath(str(index)+'_temp.txt')
                name_file_data = path_Q_crit.joinpath(str(index)+'.txt')
#                name_file_data_ready = path_Q_crit.joinpath('sequence_teste_Q'+str(index)+'_ready.json')
                
#                with open(str(name_file_data), 'w') as f:
                with open(str(name_file_data_temp), 'w') as f:
                    json.dump(time_CPU_phy.tolist(), f)
                    json.dump(Q.tolist(), f)
                os.rename(r'' + str(name_file_data_temp),r'' + str(name_file_data))
                del Q
            
            if (plt_real_time==True):
                quantiles = np.quantile(bt_MCMC[:,:,:],q=[0.025,0.975],axis=2)
                    
                ax_1.set_xlim([0, time[-1]+10])
                ax_2.set_xlim([0, time[-1]+10])
                if bt_tot.shape[1]>2:
                    ax_3.set_xlim([0, time[-1]+10])
                if bt_tot.shape[1]>3:
                    ax_4.set_xlim([0, time[-1]+10])
                if bt_tot.shape[1]>4:
                    ax_5.set_xlim([0, time[-1]+10])
                if bt_tot.shape[1]>5:
                    ax_6.set_xlim([0, time[-1]+10])
        
    #            lim = np.where((time_bt_tot<=time[-1]))[0][-1] + 1
                
                ax_1.collections.clear()
                if EV:
                    line11EV.set_data(time,particles_mean_EV[:,0])
    #                line11.set_data(time,particles_mean_EV[:,0], color=color_mean_EV)
                    ax_1.fill_between(time, quantiles_EV[0,:,0],quantiles_EV[1,:,0], color=color_quantile_EV)
                line11.set_data(time,particles_mean[:,0])
                ax_1.fill_between(time, quantiles[0,:,0],quantiles[1,:,0], color='gray')
                line13.set_data(np.array(time)[np.array(index_pf)[1:]],pos_Mes*np.ones((len(index_pf[1:]))))
                
                ax_2.collections.clear()
                if EV:
                    line21EV.set_data(time,particles_mean_EV[:,1])
    #                line21EV.set_data(time,particles_mean_EV[:,1], color=color_mean_EV)
                    if heavy_real_time_plot:
                        ax_2.fill_between(time, quantiles_EV[0,:,1],quantiles_EV[1,:,1], color=color_quantile_EV)
                line21.set_data(time,particles_mean[:,1])
                if heavy_real_time_plot:
                    ax_2.fill_between(time, quantiles[0,:,1],quantiles[1,:,1], color='gray')
                line23.set_data(np.array(time)[np.array(index_pf)[1:]],pos_Mes*np.ones((len(index_pf[1:]))))
                
                if bt_tot.shape[1]>2:
                    ax_3.collections.clear()
                    if EV:
                        line31EV.set_data(time,particles_mean_EV[:,2])
        #                line31EV.set_data(time,particles_mean_EV[:,2], color=color_mean_EV)
                        if heavy_real_time_plot:
                            ax_3.fill_between(time, quantiles_EV[0,:,2],quantiles_EV[1,:,2], color=color_quantile_EV)
                    line31.set_data(time,particles_mean[:,2])
                    if heavy_real_time_plot:
                        ax_3.fill_between(time, quantiles[0,:,2],quantiles[1,:,2], color='gray')
                    line33.set_data(np.array(time)[np.array(index_pf)[1:]],pos_Mes*np.ones((len(index_pf[1:]))))
                
                if bt_tot.shape[1]>3:
                    ax_4.collections.clear()
                    if EV:
                        line41EV.set_data(time,particles_mean_EV[:,3])
        #                line41EV.set_data(time,particles_mean_EV[:,3], color=color_mean_EV)
                        if heavy_real_time_plot:
                            ax_4.fill_between(time, quantiles_EV[0,:,3],quantiles_EV[1,:,3], color=color_quantile_EV)
                    line41.set_data(time,particles_mean[:,3])
                    if heavy_real_time_plot:
                        ax_4.fill_between(time, quantiles[0,:,3],quantiles[1,:,3], color='gray')
                    line43.set_data(np.array(time)[np.array(index_pf)[1:]],pos_Mes*np.ones((len(index_pf[1:]))))
                
                if bt_tot.shape[1]>4:
                    ax_5.collections.clear()
                    if EV:
                        line51EV.set_data(time,particles_mean_EV[:,4])
        #                line51EV.set_data(time,particles_mean_EV[:,4], color=color_mean_EV)
                        if heavy_real_time_plot:
                            ax_5.fill_between(time, quantiles_EV[0,:,0],quantiles_EV[1,:,4], color=color_quantile_EV)
                    line51.set_data(time,particles_mean[:,4])
                    if heavy_real_time_plot:
                        ax_5.fill_between(time, quantiles[0,:,4],quantiles[1,:,4], color='gray')
                    line53.set_data(np.array(time)[np.array(index_pf)[1:]],pos_Mes*np.ones((len(index_pf[1:]))))
                
                if bt_tot.shape[1]>5:
                    ax_6.collections.clear()
                    if EV:
                        line61EV.set_data(time,particles_mean_EV[:,5])
        #                line61EV.set_data(time,particles_mean_EV[:,5], color=color_mean_EV)
                        if heavy_real_time_plot:
                            ax_6.fill_between(time, quantiles_EV[0,:,5],quantiles_EV[1,:,5], color=color_quantile_EV)
                    line61.set_data(time,particles_mean[:,5])
                    if heavy_real_time_plot:
                        ax_6.fill_between(time, quantiles[0,:,5],quantiles[1,:,5], color='gray')
                    line63.set_data(np.array(time)[np.array(index_pf)[1:]],pos_Mes*np.ones((len(index_pf[1:]))))
                
                
                if plot_ref==True:
                    line14.set_data(time_bt_tot[:lim],bt_tot[:lim,0])
                    line24.set_data(time_bt_tot[:lim],bt_tot[:lim,1])
                    if bt_tot.shape[1]>2:
                        line34.set_data(time_bt_tot[:lim],bt_tot[:lim,2])
                    if bt_tot.shape[1]>3:
                        line44.set_data(time_bt_tot[:lim],bt_tot[:lim,3])
                    if bt_tot.shape[1]>4:
                        line54.set_data(time_bt_tot[:lim],bt_tot[:lim,4])
                    if bt_tot.shape[1]>5:
                        line64.set_data(time_bt_tot[:lim],bt_tot[:lim,5])
                
                
                fig.canvas.draw()
                plt.pause(0.005)    
    
    
    del bt_tronc
    
#    param['dt'] = param['dt']*n_simu
#    param['N_test'] = param['N_test']/n_simu
#    bt_MCMC = bt_MCMC[::n_simu,:,:]
#    bt_fv = bt_fv[::n_simu,:,:]
#    bt_m = bt_m[::n_simu,:,:]
#    bt_forecast_sto = bt_forecast_sto[::n_simu,:]
#    bt_forecast_deter = bt_forecast_deter[::n_simu,:]
    
#    struct_bt_MCMC = {}
#    
#    tot = {}
#    tot['mean'] = np.mean(bt_MCMC,2)
#    tot['var'] = np.var(bt_MCMC,2)
#    tot['one_realiz'] = bt_MCMC[:,:,0]
#    struct_bt_MCMC['tot'] = tot.copy()
#    
#    fv = {}
#    fv['mean'] = np.mean(bt_fv,2)
#    fv['var'] = np.var(bt_fv,2)
#    fv['one_realiz'] = bt_fv[:,:,0]
#    struct_bt_MCMC['fv'] = fv.copy()
#   
#    m = {}
#    m['mean'] = np.mean(bt_m,2)
#    m['var'] = np.var(bt_m,2)
#    m['one_realiz'] = bt_m[:,:,0]
#    struct_bt_MCMC['m'] = m.copy()
    
    
    #%%  Save 2nd results, especially I, L, C and the reconstructed Chronos

#    param = fct_name_2nd_result(param,modal_dt,reconstruction)
#    
#    
#    
##    np.savez(param['name_file_2nd_result']+'_Numpy',bt_forecast_deter=bt_forecast_deter,\
##                                                    bt_tot = bt_tot,\
##                                                    bt_forecast_sto = bt_forecast_sto,\
##                                                    param = param,\
##                                                    struct_bt_MCMC = struct_bt_MCMC,\
##                                                    bt_MCMC = bt_MCMC)  
##    
#    dict_python = {}
#    dict_python['bt_forecast_deter'] = bt_forecast_deter
#    dict_python['bt_tot'] = bt_tot
#    dict_python['bt_forecast_sto'] = bt_forecast_sto
#    dict_python['param'] = param
#    dict_python['struct_bt_MCMC'] = struct_bt_MCMC
#    dict_python['bt_MCMC'] = bt_MCMC
    
#    sio.savemat(param['name_file_2nd_result']+'_Numpy',dict_python)
    
    #%% PLOTSSSSSSSSSSSS
#    time_average = np.mean(bt_MCMC,axis=0)
#    var = np.sum(np.var(time_average,axis=1))
##    
    
    
    
    ##############################################################################################################
    #################################---TEST PLOTS---#############################################################
#    if plt_real_time == False:
    dt_tot = param['dt']
    N_test = param['N_test'] 
#    time = np.arange(1,int(int(N_test)+2),1)*float(dt_tot)
#    ref = bt_MCMC[:,:,0]
    
    particles_mean = np.mean(bt_MCMC[:,:,:],axis=2)
    particles_median = np.median(bt_MCMC[:,:,:],axis=2)
    quantiles = np.quantile(bt_MCMC[:,:,:],q=[0.025,0.975],axis=2)
    if EV:
        particles_mean_EV = np.mean(bt_forecast_EV[:,:,:],axis=2)
        particles_median_EV = np.median(bt_forecast_EV[:,:,:],axis=2)
        quantiles_EV = np.quantile(bt_forecast_EV[:,:,:],q=[0.025,0.975],axis=2)
    n_particles = bt_MCMC.shape[-1] 
#    particles_std_estimate = np.std(bt_MCMC[:,:,1:],axis=2)
#    erreur = np.abs(particles_mean-ref)


    
    
#    time_simu = np.arange(0,(bt_MCMC.shape[0])*dt_tot,dt_tot)
#    time_bt_tot = np.arange(0,(bt_MCMC.shape[0])*dt_tot,n_simu*dt_tot*3)
#    ref = bt_tot[:int(len(time_bt_tot)),:]
    for index in range(particles_mean.shape[1]):
        plt.figure(index,figsize=(12, 9))
        plt.ylim(-10, 10)
        ####        delta = 1.96*particles_std_estimate[:,index]/np.sqrt(n_particles)
        if EV:
            plt.fill_between(time,quantiles_EV[0,:,index],quantiles_EV[1,:,index],color=color_quantile_EV)
            line1_EV = plt.plot(time,particles_mean_EV[:,index],'-', \
                                color=color_mean_EV, label = 'EV particles mean')
        if plot_ref==True:
            if assimilate == 'real_data':
                plt.plot(time_bt_tot,quantiles_PIV[0,:,index],'k--',label = 'True state')
                plt.plot(time_bt_tot,quantiles_PIV[1,:,index],'k--',label = 'True state')
            else:
                plt.plot(time_bt_tot,bt_tot[:,index],'k--',label = 'True state')
            
            
        plt.fill_between(time,quantiles[0,:,index],quantiles[1,:,index],color='gray')
            
        line1 = plt.plot(time,particles_mean[:,index],'b-',label = 'Red LUM particles mean')
#            if EV:
#                line1_EV = plt.plot(time,particles_mean_EV[:,index],'-', \
#                                    color=color_mean_EV, label = 'EV particles mean')
            
#        line2 = plt.plot(time_bt_tot,ref[:,index],'k--',label = 'True state')
#        line2 = plt.plot(time_bt_tot,ref[:,index],'k--',label = 'True state')
#        line3 = plt.plot(time_simu,particles_median[:,index],'g.',label = 'particles median')
#        line4 = plt.plot(dt_tot*np.concatenate((np.zeros((1)),np.array(time_pf))),particles_estimate[:,index],'m.',label = 'PF mean estimation')
        plt.plot(np.array(time)[np.array(index_pf)[1:]],pos_Mes*np.ones((len(index_pf[1:]))),'r.')
        plt.grid()
        plt.ylabel('Chronos '+r'$b'+str(index+1)+'$'+' amplitude',fontsize=20)
        plt.xlabel('Time',fontsize=20)
        plt.legend(fontsize=15)
        file_res_mode = file_plots_res / Path(str(index+1) + '.pdf')
#            file_res_mode = file_plots_res / Path(str(index+1) + '.png')
#            file_res_mode = file_plots_res / Path(str(index+1) + '.jpg')
#            file_res_mode = file_plots_res / str(index) + '.png'
#            plt.savefig(file_res_mode)
        plt.savefig(file_res_mode,dpi=200 )
#            plt.savefig(file_res_mode,dpi=500 )
#            plt.savefig(file_res_mode,quality = 95)

    
#        print(time_bt_tot.shape)
#        print(time.shape)
    
    dict_python = {}
    dict_python['param'] = param
    dict_python['time_bt_tot'] = time_bt_tot
    dict_python['bt_tot'] = bt_tot
    dict_python['quantiles_bt_tot'] = quantiles_PIV
    dict_python['dt_PIV'] = dt_PIV
    dict_python['index_pf'] = index_pf
    dict_python['time_DA'] = np.array(time)[np.array(index_pf)[1:]]
    dict_python['Re'] = Re
    dict_python['time'] = time
    dict_python['particles_mean'] = particles_mean
    dict_python['particles_median'] = particles_median
    dict_python['bt_MCMC'] = bt_MCMC
    dict_python['quantiles'] = quantiles
    if EV:
        dict_python['particles_mean_EV'] = particles_mean_EV
        dict_python['particles_median_EV'] = particles_median_EV
        dict_python['bt_forecast_EV'] = bt_forecast_EV
        dict_python['quantiles_EV'] = quantiles_EV
    file_res = file_plots_res / Path('chronos.mat')
    sio.savemat(file_res,dict_python)
    
    
    
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
        
    ############################ Construct the path to select the model constants I,L,C,pchol and etc.
    
            
        
        
        
#    '''
#    Now we have the estimation of the Chronos, therefore it's necessary to estabilish some measures to compare with the PIV measured flow. 
#    '''       
        
        
#    Champ_smoothed = Hpiv_Topos @ np.concatenate((particles_mean,np.ones((particles_mean.shape[0],1))),axis=1).T
    
#    topos_Fx = Hpiv_Topos[:14948,:]
#    topos_Fy = Hpiv_Topos[14948:,:]
#    
#    
#    delta_space = coordinates_x_PIV[1]-coordinates_x_PIV[0]
#    nb_x = topos_new_coordinates.shape[0]
#    nb_y = topos_new_coordinates.shape[1]
#    curl_Z = calculate_rotational(topos_Fx,topos_Fy,delta_space,nb_x,nb_y)
#    
#    Champ_smoothed = np.reshape(curl_Z,(int(curl_Z.shape[0]*curl_Z.shape[1]),curl_Z.shape[2]),order='F') @ np.concatenate((particles_mean,np.ones((particles_mean.shape[0],1))),axis=1).T
#    
#    Champ_smoothed_reshaped = np.reshape(Champ_smoothed,(nb_x-2,nb_y-2,Champ_smoothed.shape[-1]),order='F')   
#    
#    
#    for i in range(Champ_smoothed_reshaped.shape[-1]):
#        plt.imshow(Champ_smoothed_reshaped[:,:,i].T)
#        
#        plt.pause(0.1)
#       
#    
#    PIV_Fx = matrix_data_PIV_all_data[:,:,0].T
#    PIV_Fy = matrix_data_PIV_all_data[:,:,1].T
#    curl_Z_PIV = calculate_rotational(PIV_Fx,PIV_Fy,delta_space,nb_x,nb_y)
    
#    for i in range(curl_Z_PIV.shape[-1]):
#        plt.imshow(curl_Z_PIV[:,:,i].T)
#        plt.pause(0.001)
    
    
        

#################################   SAVE 
#    
#    dict_python = {}
#    dict_python['particles_mean'] = particles_mean[time_pf,:]
#    dict_python['MX'] = param['MX']
##    dict_python['index_filter'] = time_pf
#        
#    name_file_data = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath(str(particles_mean.shape[1])+'modes_particle_mean')
#    sio.savemat(str(name_file_data)+'_Numpy',dict_python)
    ##############################################################################################################
    ##############################################################################################################
#    dict_python = {}
#    dict_python['obs'] = observations_to_save
#    dict_python['bt_MCMC'] = particles_mean
#    dict_python['particles'] = bt_MCMC
#    dict_python['iters'] = param['N_test']
#    dict_python['dt'] = param['dt']
#    dict_python['bt_tot'] = bt_tot
#    dict_python['n_simu'] = n_simu
#    dict_python['param'] = param
#    dict_python['index_of_filtering'] = index_of_filtering
#    name_file_data = Path(__file__).parents[3].joinpath('test').joinpath('data_to_benchmark'+str(nb_modes)+'_bruit_'+str(beta_1)+'reynolds300')
#    sio.savemat(str(name_file_data),dict_python)
    
    # Time of execution
    print(choice_n_subsample)
    print('\n')
    print('Time of execution of PF & Red LUM for ' + \
          str(SECONDS_OF_SIMU) + ' s of simulation:')
    print('\n')
    print(str(time_exe) + ' s')
    print('\n')
    if EV:
        print('Time of execution of PF & random EV for ' + \
              str(SECONDS_OF_SIMU) + ' s of simulation:')
        print('\n')
        print(str(time_exe_EV) + ' s')
        print('\n')
        print('Ratio speed :')
        print('\n')
        print(str(100*time_exe_EV/time_exe) + ' %')
        print('\n')
    
    
    del C_deter 
    del C_sto 
    del L_deter 
    del L_sto 
    del I_deter 
    del I_sto

    
    
    
    return 0 #var
    
    #%%
    
    
if __name__ == '__main__':
    
    nb_modes = 8
    modal_dt = False
    threshold = 1e-05
    adv_corrected = False
    type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated'
    no_subampl_in_forecast = False
    nb_period_test = math.nan
    reconstruction = False
    #nb_modes,threshold,type_data,nb_period_test,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt
    result = main_from_existing_ROM(nb_modes,threshold,type_data,nb_period_test,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt)
    
    
    
    
    
#    spectrum = np.zeros(bt_tot.shape)
#    for i in range(nb_modes):
#        spec = np.power(np.abs(np.fft.fft(bt_tot[:,i])),2)
#        spectrum[:,i] = spec
#    
#    spec1 = spectrum[:,0]
#    
#    
#    spec1 = spec1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    