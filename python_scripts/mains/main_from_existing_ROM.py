# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:17:08 2019

@author: matheus.ladvig
"""
import math
import os
from convert_mat_to_python import convert_mat_to_python
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



def estimate_lambda(time_obs):
    
    b_power = np.power(time_obs,2)
    lambda_values = np.sum(b_power,axis=0)/time_obs.shape[0]


    return lambda_values







def main_from_existing_ROM(nb_modes,threshold,type_data,nb_period_test,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt):#nb_modes,threshold,type_data,nb_period_test,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt):
    
#    --Load simulation results
#    --Estimate modal time step by Shanon 
#    --Compare it with modal Eddy Viscosity ROM and tuned version of the loaded results.
    
    
    # Parameters choice
    param_ref = {}
    param_ref['n_simu'] = 5
    param_ref['N_particules'] = 5
    
    #%%  Parameters already chosen
    
    #   !!!! Do not modify the following lines  !!!!!!
#    coef_correctif_estim = {}
#    coef_correctif_estim['learn_coef_a'] = True # true or false
#    coef_correctif_estim['type_estim'] = 'vector_b' #  'scalar' 'vector_z' 'vector_b' 'matrix'
#    coef_correctif_estim['beta_min'] =  math.inf # -inf 0 1
    
    
    current_pwd = Path(__file__).parents[1]
    folder_results = current_pwd.parents[0].joinpath('resultats').joinpath('current_results')
    folder_data = current_pwd.parents[0].joinpath('data')
    
    
    param_ref['folder_results'] = str(folder_results)
    param_ref['folder_data'] = str(folder_data)
    
    
    modal_dt_ref = modal_dt
     
    
    
    #%% Get data
    
    # On which function the Shanon ctriterion is used
    test_fct = 'b'  # 'b' is better than db
    a_t = '_a_cst_'
    
    
    file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
            a_t + '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' + str(threshold) + \
            'fct_test_' + test_fct
    
    
    
              
    var_exits =  'var' in locals() or 'var' in globals()
    period_estim = 'period_estim' in locals() or 'period_estim' in globals()
    
    if var_exits == True and period_estim == True:
        file = file + '_p_estim_' + str(period_estim);
    
    file = file + '_fullsto'
    
    if not adv_corrected:
        file = file + '_no_correct_drift'
    
    
    file = file + '.mat'
    
    print(file)
    
    
    file_res = folder_results / Path(file)
    
    # The function creates a dictionary with the same structure as the Matlab Struct in the path file_res
    I_sto,L_sto,C_sto,I_deter,L_deter,C_deter,plot_bts,pchol_cov_noises,bt_tot,param = convert_mat_to_python(str(file_res))
    
    param['decor_by_subsampl']['no_subampl_in_forecast'] = no_subampl_in_forecast
    
    
    
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
    
    
    #%% Choice of modal time step
    #TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
#    modal_dt = True
    ##############################################################################
    if modal_dt == True:
#        rate_dt,ILC_a_cst,pchol_cov_noises = fct_cut_frequency_2_full_sto(bt_tot,ILC_a_cst,param,pchol_cov_noises,modal_dt)
        rate_dt, ILC_a_cst,pchol_cov_noises = fct_cut_frequency_2_full_sto(bt_tot,ILC_a_cst,param,pchol_cov_noises,modal_dt)
    else:
        ILC_a_cst['modal_dt'] = ILC_a_cst['tot']
    
    
    #%% Do not temporally subsample, in order to prevent aliasing in the results
    
    if not reconstruction:
        
#        if exists: 
        current_pwd = Path(__file__).parents[1]
        if param['nb_period_test'] is math.nan:
            name_file_data = current_pwd.parents[1].joinpath('data').joinpath( type_data + '_' + str(nb_modes) + '_modes' + '_threshold_' + str(param['decor_by_subsampl']['spectrum_threshold']) + \
                                               '_nb_period_test_' + 'NaN' + '_Chronos_test_basis.mat')
        else:
            name_file_data = current_pwd.parents[1].joinpath('data').joinpath( type_data + '_' + str(nb_modes) + '_modes' + '_threshold_' + str(param['decor_by_subsampl']['spectrum_threshold']) + \
                                               '_nb_period_test_' + str(param['nb_period_test']) + '_Chronos_test_basis.mat')
        
           
       
            
        if name_file_data.exists():
            mat = hdf5storage.loadmat(str(name_file_data))
            bt_tot = mat['bt']
            truncated_error2 = mat['truncated_error2']
            param['truncated_error2'] = truncated_error2
            
            
            
        else:
            print('ERROR: File does not exist ', str(name_file_data))
            return 0
        
        if param['big_data'] == True:
            print('Test basis creation done')
        
        #Test basis creation
        
    
    #%% Time integration of the reconstructed Chronos b(t)
    
    param['folder_results'] = param_ref['folder_results']
    param['N_particules'] = param_ref['N_particules']
    n_simu = param_ref['n_simu']
    param['N_tot'] = bt_tot.shape[0]
    param['N_test'] = param['N_tot'] - 1
    bt_tot = bt_tot[:param['N_test'] + 1,:] #  reference Chronos
    bt_tronc=bt_tot[0,:][np.newaxis] # Initial condition
    
    param['dt'] = param['dt']/n_simu
    param['N_test'] = param['N_test'] * n_simu
    
    
#    Reconstruction in the deterministic case
    bt_forecast_deter = bt_tronc
    for index in range(param['N_test']):
        bt_forecast_deter = np.vstack((bt_forecast_deter, evol_forward_bt_RK4(ILC_a_cst['deter']['I'],ILC_a_cst['deter']['L'],ILC_a_cst['deter']['C'],param['dt'],bt_forecast_deter)))
    
#    Reconstruction in the stochastic case
    

    bt_forecast_sto = bt_tronc
    for index in range(param['N_test']):
        bt_forecast_sto = np.vstack((bt_forecast_sto,evol_forward_bt_RK4(ILC_a_cst['modal_dt']['I'],ILC_a_cst['modal_dt']['L'],ILC_a_cst['modal_dt']['C'],param['dt'],bt_forecast_sto)))
    
    
    
    
#    Reconstruction in the stochastic case
    
    bt_MCMC = np.tile(bt_tronc.T,(1,1,param['N_particules']))
    bt_fv = bt_MCMC.copy()
    bt_m = np.zeros((1,int(param['nb_modes']),param['N_particules']))
    
    iii_realization = np.zeros((param['N_particules'],1))
    for index in range(param['N_test']):
        ##### Regarder evol pour efacer les matrices deterministes
        val0,val1,val2 = evol_forward_bt_MCMC(ILC_a_cst['modal_dt']['I'],\
                                                        ILC_a_cst['modal_dt']['L'],\
                                                        ILC_a_cst['modal_dt']['C'],\
                                                        pchol_cov_noises,param['dt'],\
                                                        bt_MCMC[-1,:,:],bt_fv[-1,:,:],\
                                                        bt_m[-1,:,:])
        
        #########################################----------------------#############################################
        #########################################--PARTICLE FILTERING--#############################################
        obs = val0[0,:,0][...,np.newaxis]
        particles = val0[0,:,1:]
        #lambda_values = estimate_lambda(bt_MCMC[:,:,0])
        lambda_values = param['lambda'][:,0]
        particles = particle_filter(particles,obs,lambda_values)
        val0 = np.hstack((obs,particles))[np.newaxis,...]
        #############################################################################################################
        #############################################################################################################
        
        bt_MCMC = np.concatenate((bt_MCMC,val0),axis=0)    
        bt_fv   = np.concatenate((bt_fv,val1),axis=0)
        bt_m    = np.concatenate((bt_m,val2),axis=0)
        
        
        iii_realization = np.any(np.logical_or(np.isnan(bt_MCMC[index+1,:,:]),np.isinf(bt_MCMC[index+1,:,:])),axis = 0)[...,np.newaxis]
        
        if np.any(iii_realization):
            if np.all(iii_realization):
                print('WARNING: All realization of the simulation have blown up.')
                
                if index < param['N_test']:
                    val_nan = np.full([int(param['N_test']-index), param['nb_modes'], param['N_particules']], np.nan)
                    bt_MCMC = np.concatenate((bt_MCMC,val_nan),axis=0)    
                    bt_fv   = np.concatenate((bt_fv,val_nan),axis=0)
                    bt_m    = np.concatenate((bt_m,val_nan),axis=0)
                
                break 
            
            
        
            nb_blown_up = np.sum(iii_realization)
            print('WARNING: '+ str(nb_blown_up)+' realizations have blown up and will be replaced.')
            good_indexes = np.where((np.logical_not(iii_realization) == True))[0]
            bt_MCMC_good = bt_MCMC[-1,:, good_indexes].T
            bt_fv_good = bt_fv[-1,:, good_indexes].T
            bt_m_good = bt_m[-1,:, good_indexes].T
            
            
            bt_MCMC_good = bt_MCMC_good[np.newaxis,...]
            bt_fv_good = bt_fv_good[np.newaxis,...]
            bt_m_good = bt_m_good[np.newaxis,...]
                
            
            rand_index =  np.random.randint(0, param['N_particules'] - nb_blown_up, size=(nb_blown_up))
            
#            if rand_index.shape == (1,1):
#                rand_index = rand_index[0,0]
                
                
            bad_indexes = np.where((iii_realization == True))[0]
            bt_MCMC[-1,:, bad_indexes] = bt_MCMC_good[0,:, rand_index]  
            bt_fv[-1,:, bad_indexes] = bt_fv_good[0,:, rand_index]
            bt_m[-1,:, bad_indexes] = bt_m_good[0,:, rand_index]
    
            del bt_MCMC_good 
            del rand_index 
            del nb_blown_up 
            del iii_realization     
    
    
    
    del bt_tronc
    
    param['dt'] = param['dt']*n_simu
    param['N_test'] = param['N_test']/n_simu
    bt_MCMC = bt_MCMC[::n_simu,:,:]
    bt_fv = bt_fv[::n_simu,:,:]
    bt_m = bt_m[::n_simu,:,:]
    bt_forecast_sto = bt_forecast_sto[::n_simu,:]
    bt_forecast_deter = bt_forecast_deter[::n_simu,:]
    
    struct_bt_MCMC = {}
    
    tot = {}
    tot['mean'] = np.mean(bt_MCMC,2)
    tot['var'] = np.var(bt_MCMC,2)
    tot['one_realiz'] = bt_MCMC[:,:,0]
    struct_bt_MCMC['tot'] = tot.copy()
    
    fv = {}
    fv['mean'] = np.mean(bt_fv,2)
    fv['var'] = np.var(bt_fv,2)
    fv['one_realiz'] = bt_fv[:,:,0]
    struct_bt_MCMC['fv'] = fv.copy()
   
    m = {}
    m['mean'] = np.mean(bt_m,2)
    m['var'] = np.var(bt_m,2)
    m['one_realiz'] = bt_m[:,:,0]
    struct_bt_MCMC['m'] = m.copy()
    
    
    #%%  Save 2nd results, especially I, L, C and the reconstructed Chronos

    param = fct_name_2nd_result(param,modal_dt,reconstruction)
    
    
    
#    np.savez(param['name_file_2nd_result']+'_Numpy',bt_forecast_deter=bt_forecast_deter,\
#                                                    bt_tot = bt_tot,\
#                                                    bt_forecast_sto = bt_forecast_sto,\
#                                                    param = param,\
#                                                    struct_bt_MCMC = struct_bt_MCMC,\
#                                                    bt_MCMC = bt_MCMC)  
#    
    dict_python = {}
    dict_python['bt_forecast_deter'] = bt_forecast_deter
    dict_python['bt_tot'] = bt_tot
    dict_python['bt_forecast_sto'] = bt_forecast_sto
    dict_python['param'] = param
    dict_python['struct_bt_MCMC'] = struct_bt_MCMC
    dict_python['bt_MCMC'] = bt_MCMC
    
    sio.savemat(param['name_file_2nd_result']+'_Numpy',dict_python)
    
    
    del C_deter 
    del C_sto 
    del L_deter 
    del L_sto 
    del I_deter 
    del I_sto

    
    
    
    return 0
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    