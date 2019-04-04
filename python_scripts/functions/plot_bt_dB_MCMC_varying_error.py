# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:51:29 2019

@author: matheus.ladvig
"""
import numpy as np


def plot_bt_dB_MCMC_varying_error(param,zzz,zzz,zzz, zzz, bt_forecast_deter,bt_forecast_MEV,bt_forecast_sto,zzz,bt_tot,\
                                  struct_bt_MCMC,bt_MCMC):
    
    
    logscale = FDalse
    LineWidth = 1
    FontSize = 10
    FontSizeTtitle = 11
    width=1
    height=0.7
    
    height = height*3/2
    
    X0 = np.array([0,0])


    plot_deter = param['plot']['plot_deter']
    plot_EV = param['plot']['plot_EV']
    plot_tuned = param['plot']['plot_tuned']
    plot_modal_dt = param['plot_modal_dt']
    
    param['param']['nb_modes'] =  bt_tot.shape[1]
    
    dt_tot = param['dt']
    
    
    if 'N_tot' in param.keys():
        N_tot = param['N_tot']
        N_test = param['N_test']
    else:
        N_tot = 300
        N_test = 299
        
    bt_MCMC = bt_MCMC[:int(N_test),:,:]
    struct_bt_MCMC['tot']['mean'] = struct_bt_MCMC['tot']['mean'][:int(N_test),:]
    struct_bt_MCMC['tot']['var'] = struct_bt_MCMC['tot']['var'][:int(N_test),:]
    struct_bt_MCMC['tot']['one_realiz'] = struct_bt_MCMC['tot']['one_realiz'][:int(N_test),:]
    
    bt_tot = bt_tot[:int(N_test),:]
    bt_forecast_deter = bt_forecast_deter[:int(N_test),:]
    bt_forecast_MEV = bt_forecast_MEV[:int(N_test),:]
    bt_sans_coef1 = bt_sans_coef1[:int(N_test),:]
    if np.logical_not( param['reconstruction']):
        param['truncated_error2'] = param['truncated_error2'][:int(N_test),:]
    
    N_test = N_test-1
    
    dt_tot = param['dt']
    N_time_final = N_tot
    time = [:int(N_test+1)]*dt_tot
    time_ref = time
    
    
    #%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    