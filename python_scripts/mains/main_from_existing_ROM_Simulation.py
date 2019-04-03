# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:30:51 2019

@author: matheus.ladvig
"""
import math
import numpy as np
import hdf5storage
from pathlib import Path


def main_from_existing_ROM_Simulation(type_data,nb_modes,threshold,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt):
    
    #    Load simulation results, estimate modal time step by Shanon
    #    and compare it with modal Eddy Viscosity ROM and
    #    tuned version of the loaded results
        
        
    #     Plots to do
        
    plot_deterministic=true   #deterministic POD-Galerkin
    plot_EV = True            #estimated  Eddy Visocvity
    plot_tuned = False        #estimated corrective coefficients
    plot_each_mode = False;
    test_fct = 'b'            # 'b' is better than db
    
    
    #%% Parameters already chosen
    # Do not modify the following lines
    
    coef_correctif_estim = {}
    
     
    coef_correctif_estim['learn_coef_a'] = True  # true false
    coef_correctif_estim['type_estim'] = 'vector_b' # 'scalar' 'vector_z' 'vector_b' 'matrix'
    coef_correctif_estim['beta_min'] = - math.inf  # -inf 0 1
    coef_correctif_estim['nb_modes_used'] = nb_modes  # % 2 eval('nb_modes') for learning the coefficient
    
    
#    folder_results 
    
    
    folder_results = Path(__file__).parents[2].joinpath('resultats').joinpath('current_results')
    folder_data = Path(__file__).parents[3].joinpath('data')
    
    
    param_ref2 = {}
    param_ref2['folder_results'] = str(folder_results)
    param_ref2['folder_data'] = str(folder_data)
    
    modal_dt_ref = modal_dt
    folder_results_ref = folder_results
    folder_data_ref = folder_data
    
    #%% Get data
    
    a_t = '_a_cst_'
    
    file_res_2nd_res = '2ndresult_' + type_data + '_' + str(int(nb_modes)) + '_modes_' + \
            a_t + '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' + str(threshold) + \
            'fct_test_' + test_fct
            
            
            
    file_res_2nd_res = file_res_2nd_res + '_fullsto'
    
    if modal_dt == 1:
        file_res_2nd_res = file_res_2nd_res + '_modal_dt'
    elif modal_dt == 2:
        file_res_2nd_res = file_res_2nd_res + '_real_dt'
        
        
    if np.logical_not(adv_corrected):
        file_res_2nd_res = file_res_2nd_res + '_no_correct_drift'

    if no_subampl_in_forecast:
        file_res_2nd_res = file_res_2nd_res + '_no_subampl_in_forecast'

    if reconstruction:
        file_res_2nd_res = file_res_2nd_res + '_reconstruction'
    
    file_res_2nd_res = folder_results / Path(file_res_2nd_res + '.mat') 
    
    mat = hdf5storage.loadmat(str(file_res_2nd_res))
    
    
    if reconstruction:
        param['reconstruction'] = True
    else:
        param['reconstruction'] = False
    
    
    folder_results = folder_results_ref
    param['folder_results'] = str(folder_results)
    folder_data = folder_data_ref
    param['folder_data'] = str(folder_data)
    modal_dt = modal_dt_ref

    param['decor_by_subsampl']['test_fct'] = test_fct

    folder_data = param_ref2['folder_data']
    folder_results = param_ref2['folder_results']
    
    big_data = False
    plot_bts = True
    
    
    param['folder_data'] = folder_data
    param['folder_results'] = folder_results
    param['big_data'] = big_data
    param['plot_bts'] = plot_bts
    param['coef_correctif_estim'] = coef_correctif_estim.copy()
        
    param['folder_results'] = param_ref2['folder_results']
    
    
    
    
    
    
    struct_bt_MCMC['tot']['one_realiz'] = bt_MCMC[:,:,1]
    
    #%% Eddy viscosity solutions
    
    if plot_EV:
        param['plot']['plot_EV'] = plot_EV
        file_EV =  'EV_result_' + param['type_data'] + '_' + str(int(param['nb_modes'])) + '_modes'
        file_EV = file_EV + '.mat'
        
        folder_results = Path(param['folder_results']).joinpath(file_EV)
        
        
        mat = hdf5storage.loadmat(str(file_res_2nd_res))
#        load(file_EV,'param_deter',...
#            'bt_forecast_MEV','bt_forecast_EV','bt_forecast_NLMEV');
        
        
        bt_forecast_MEV = bt_forecast_EV
       
        del bt_forecast_EV
        bt_forecast_MEV = bt_forecast_MEV[::param['decor_by_subsampl']['n_subsampl_decor'],:]
        
        
        
    
    #%% Plots of the reconstructed Chronos
    
    if plot_bts:
        param['folder_data'] =param_ref2['folder_data'] 
        param['plot']['plot_deter'] = plot_deterministic
        param['plot']['plot_EV'] = plot_EV
        param['plot']['plot_tuned'] = plot_tuned
        param['plot_modal_dt'] = False
        
        zzz = np.zeros(bt_tot.shape)
    
        if  np.logical_not(param['plot']['plot_EV']):
            bt_forecast_MEV = zzz
        
        
        
        plot_bt_MCMC(param,bt_tot,bt_tot,bt_tot, bt_tot, bt_forecast_deter,bt_forecast_MEV, \
                     bt_forecast_sto,bt_forecast_sto,bt_tot,struct_bt_MCMC)
        
        
        idx_min_error, idx_min_err_tot = plot_bt_dB_MCMC_varying_error(param,zzz,zzz,zzz, zzz, bt_forecast_deter,\
                                                                       bt_forecast_MEV,bt_forecast_sto,zzz,bt_tot,\
                                                                       struct_bt_MCMC,bt_MCMC)
        
        
#        save(file_res_2nd_res,'idx_min_error','idx_min_err_tot','-append')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    
    

    
    
    
    