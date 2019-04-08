# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:30:51 2019

@author: matheus.ladvig
"""
import math
import numpy as np
import hdf5storage
from pathlib import Path
import sys
path_functions = Path(__file__).parents[1].joinpath('functions')
sys.path.insert(0, str(path_functions))
from plot_bt_MCMC import plot_bt_MCMC
from plot_bt_dB_MCMC_varying_error import plot_bt_dB_MCMC_varying_error

def load_data_from_results(file_res_2nd_res):
    variables = hdf5storage.loadmat(file_res_2nd_res)
    param = variables['param']
    struct_bt_MCMC = variables['struct_bt_MCMC']
    bt_MCMC = variables['bt_MCMC']
    bt_forecast_sto = variables['bt_forecast_sto']
    bt_tot = variables['bt_tot']
    bt_forecast_deter = variables['bt_forecast_deter']
    
    
    
    
    param_dict = {}
    
    # Loop to run all the dtypes in the data
    for name in param.dtype.names :
        
        
        # If the dtype has only one information we take his name and create a key for this information
        if param[0][name].dtype.names == None:
            if param[0][name].shape == (1,1):
                param_dict[name] = param[0][name][0,0]
            else:
                param_dict[name] = param[0][name]
        # If not, we need to create a dict that will be resposnsible to stock all fields of this structure
        
        else:
            aux_dict = {}
            for name2 in param[0][name].dtype.names:
                if param[0][name][name2].shape == (1,1,1):
                    aux_dict[name2] = param[0][name][name2][0,0,0]
                else:
                    aux_dict[name2] = param[0][name][name2]
            
            param_dict[name] = aux_dict
    
    struct_bt_MCMC_dict = {}
    
    for name in struct_bt_MCMC.dtype.names :
        
        # If the dtype has only one information we take his name and create a key for this information
        if struct_bt_MCMC[0][name].dtype.names == None:
            if struct_bt_MCMC[0][name].shape == (1,1):
                struct_bt_MCMC_dict[name] = struct_bt_MCMC[0][name][0,0]
            else:
                struct_bt_MCMC_dict[name] = struct_bt_MCMC[0][name]
        # If not, we need to create a dict that will be resposnsible to stock all fields of this structure
        
        else:
            aux_dict = {}
            for name2 in struct_bt_MCMC[0][name].dtype.names:
                if struct_bt_MCMC[0][name][name2].shape == (1,1,1):
                    aux_dict[name2] = struct_bt_MCMC[0][name][name2][0,0,0]
                else:
                    aux_dict[name2] = struct_bt_MCMC[0][name][name2]
            
            struct_bt_MCMC_dict[name] = aux_dict
    
    
    return bt_forecast_deter,bt_tot,bt_forecast_sto,param_dict.copy(),struct_bt_MCMC_dict.copy(),bt_MCMC
    
def load_file_EV(file_res_2nd_res):
    variables = hdf5storage.loadmat(file_res_2nd_res)
    
    bt_forecast_MEV = variables['bt_forecast_MEV']
    bt_forecast_EV = variables['bt_forecast_EV']
    bt_forecast_NLMEV = variables['bt_forecast_NLMEV']
    
    param_deter = variables['param_deter']
    
    param_dict = {}
    
    # Loop to run all the dtypes in the data
    for name in param_deter.dtype.names :
        
        if param_deter[0][name][0][0].dtype.names == None:
            param_dict[name] = param_deter[0][name][0]
            
#            if param_deter[0][name].shape == (1,1):
#                param_dict[name] = param_deter[0][name][0,0]
#            else:
#                param_dict[name] = param_deter[0][name]
        # If not, we need to create a dict that will be resposnsible to stock all fields of this structure
        
        else:
            aux_dict = {}
            for name2 in param_deter[0][name][0][0].dtype.names:
                
                    aux_dict[name2] = param_deter[0][name][0][0][name2][0]
                
#            
            param_dict[name] = aux_dict
    
    
    return param_dict.copy(),bt_forecast_MEV,bt_forecast_EV,bt_forecast_NLMEV
    
    
    
def main_from_existing_ROM_Simulation(type_data,nb_modes,threshold,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt,plt,\
                                      varying_error_figure,nb_subplot_cols,current_subplot):
    
    #    Load simulation results, estimate modal time step by Shanon
    #    and compare it with modal Eddy Viscosity ROM and
    #    tuned version of the loaded results
        
        
    #     Plots to do
        
    plot_deterministic = True   #deterministic POD-Galerkin
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
    
    bt_forecast_deter,bt_tot,bt_forecast_sto,param,struct_bt_MCMC,bt_MCMC = load_data_from_results(str(file_res_2nd_res))
    
    
    
    
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
    
    
    
    
    
    
    struct_bt_MCMC['tot']['one_realiz'] = bt_MCMC[:,:,0]
    
    #%% Eddy viscosity solutions
    
    if plot_EV:
        plot = {}
        plot['plot_EV'] = plot_EV
        param['plot'] =  plot
        file_EV =  'EV_result_' + param['type_data'] + '_' + str(int(param['nb_modes'])) + '_modes'
        file_EV = file_EV + '.mat'
        
        folder_results = Path(param['folder_results']).joinpath(file_EV)
        
        param_deter,bt_forecast_MEV,bt_forecast_EV,bt_forecast_NLMEV = load_file_EV(str(folder_results))

        
        
        bt_forecast_MEV = bt_forecast_EV
       
        del bt_forecast_EV
        bt_forecast_MEV = bt_forecast_MEV[::int(param['decor_by_subsampl']['n_subsampl_decor']),:]
        
        
        
    
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
                     bt_forecast_sto,bt_forecast_sto,bt_tot,struct_bt_MCMC,plt)
        
        
        idx_min_error, idx_min_err_tot = plot_bt_dB_MCMC_varying_error(param,zzz,zzz,zzz, zzz, bt_forecast_deter,\
                                                                       bt_forecast_MEV,bt_forecast_sto,zzz,bt_tot,\
                                                                       struct_bt_MCMC,bt_MCMC,plt,varying_error_figure,\
                                                                       nb_subplot_cols,current_subplot)
        
        
#        save(file_res_2nd_res,'idx_min_error','idx_min_err_tot','-append')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    pass
    

    
    
    
    