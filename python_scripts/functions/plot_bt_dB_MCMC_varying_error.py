# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:51:29 2019

@author: matheus.ladvig
"""
import numpy as np
import os
#import matplotlib.pyplot as plt
import hdf5storage
from pathlib import Path


def load_mat(filename,variable_to_load):
    
    variables = hdf5storage.loadmat(filename)
    
    
    return variables[variable_to_load]

def load_param_c(param):
    
    
    param_dict = {}
    
    # Loop to run all the dtypes in the data
    for name in param.dtype.names :
        
        # If the dtype has only one information we take his name and create a key for this information
        if param[0][name][0].dtype.names == None:
            if param[0][name].shape == (1,1):
                param_dict[name] = param[0][name][0,0]
            else:
                param_dict[name] = param[0][name][0]
        # If not, we need to create a dict that will be resposnsible to stock all fields of this structure
        
        else:
            aux_dict = {}
            for name2 in param[0][name][0].dtype.names:
                if param[0][name][0][name2].shape == (1,1):
                    aux_dict[name2] = param[0][name][0][name2][0,0]
                else:
                    aux_dict[name2] = param[0][name][0][name2][0]
            
            param_dict[name] = aux_dict
    
    
    
    
    
    
    
    return param_dict.copy()
    




def plot_bt_dB_MCMC_varying_error(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,bt_forecast_sto_a_cst_modal_dt,\
                                  bt_forecast_sto_a_NC_modal_dt, bt_forecast_deter,bt_forecast_MEV,bt_sans_coef1,\
                                  bt_sans_coef2,bt_tot,struct_bt_MCMC,bt_MCMC,plt,varying_error_figure,nb_subplot_cols,current_subplot):
    
    
    logscale = False
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
    
    
#    param['param']['nb_modes'] =  bt_tot.shape[1]
    
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
    time = np.arange(1,int(N_test + 2),1)*float(dt_tot)
    time_ref = time
    
    
    #%%
    
    if param['type_data'] == 'LES_3D_tot_sub_sample_blurred':
        nrj_varying_tot = 18.2705
        current_pwd = Path(__file__).parents[1]
        folder_results = current_pwd.parents[0].joinpath('resultats').joinpath('nrj_mean_LES3900')
        variables = hdf5storage.loadmat(str(folder_results))
        
    
    elif (param['type_data'] == 'inc3D_Re300_40dt_blocks_truncated') or (param['type_data'] == 'inc3D_Re300_40dt_blocks'):
        folder_Path = Path(param['folder_results']).joinpath('_nrj_mU_' + param['type_data'] + '0.mat')
#        param['file_nrj_mU'] =  param['folder_results'] + '_nrj_mU_' + param['type_data'] + '0.mat'
#        load(param.file_nrj_mU, 'nrj_mU')
        variables = hdf5storage.loadmat(str(folder_Path))
        norm0 = nrj_mU
#        param['name_file_pre_c'] = param['folder_data'] + param['type_data'] + '_pre_c'
        param_ref = param
        
        folder_Path = Path(param['folder_data']).joinpath(param['type_data'] + '_pre_c')
        param['name_file_pre_c'] = str(folder_Path)
        variables = hdf5storage.loadmat(str(folder_Path))
#        load(param.name_file_pre_c,'c','param')
        param_c = param
        param = param_ref
        del param_ref
        
        c = c*np.prod(param.dX)/param_c.N_tot;
        
        nrj_varying_tot=np.matrix.trace(c)
        
        
        
        
    
    
    elif param['type_data'] ==  'inc3D_Re3900_blocks' or param['type_data'] == 'inc3D_Re3900_blocks119':
        current_pwd = Path(__file__).parents[1]
        folder_results = current_pwd.parents[0].joinpath('resultats').joinpath('nrj_mean_DNS3900blurred')
        variables = hdf5storage.loadmat(str(folder_results))
        
        
        
        folder_Path = Path(param['folder_data']).joinpath(param['type_data'] + '_pre_c')
        param['name_file_pre_c'] = str(folder_Path)
        param_ref = param
#        load(param.name_file_pre_c,'c','param');
        variables = hdf5storage.loadmat(str(folder_Path))
        param_c = param
        param = param_ref
        del param_ref
        c = c*np.prod(param['dX'])/param_c['N_tot']
        nrj_varying_tot = np.matrix.trace(c)
    
    
    
    
    elif param['type_data'] == 'incompact3d_wake_episode3_cut':
        
        folder_Path = Path(param['folder_data']).joinpath('_nrj_mU_' + param['type_data'] + '.mat')
        param['file_nrj_mU'] = str(folder_Path)
        variables = hdf5storage.loadmat(str(folder_Path))
        
#        load(param.file_nrj_mU, 'nrj_mU');
        norm0 = nrj_mU
        
        folder_Path = Path(param['folder_data']).joinpath(param['type_data'] + '_pre_c')
        param['name_file_pre_c'] = str(folder_Path)
        param_ref = param
        
        variables = hdf5storage.loadmat(str(folder_Path))
#        load(param.name_file_pre_c,'c','param');
        param_c = param
        param = param_ref
        del param_ref
        c = c*np.prod(param['dX'])/param_c['N_tot']
        nrj_varying_tot = np.matrix.trace(c)
        
    
    
    else:  
        
        folder_Path = Path(param['folder_results']).joinpath('_nrj_mU_' + param['type_data'] + '0.mat')
        param['file_nrj_mU'] = str(folder_Path)
        
        
        if os.path.isfile(param['file_nrj_mU']):
            nrj_mU = load_mat(param['file_nrj_mU'],'nrj_mU')
            
#            
        else:
#    
            type_data_mU = param['type_data'] + str(0)
            folder_Path = Path(param['folder_data']).joinpath(type_data_mU + '_U_centered')
            param['name_file_mU'] = str(folder_Path)
            
            variables = hdf5storage.loadmat(str(folder_Path))
            m_U = variables['m_U']
#            
            names_list = ['incompact3d_wake_episode3_cut_truncated',\
                            'incompact3D_noisy2D_40dt_subsampl',\
                            'incompact3D_noisy2D_40dt_subsampl_truncated',\
                            'inc3D_Re300_40dt_blocks',\
                            'inc3D_Re300_40dt_blocks_truncated',\
                            'inc3D_Re300_40dt_blocks_test_basis',\
                            'inc3D_Re3900_blocks',\
                            'inc3D_Re3900_blocks_truncated',\
                            'inc3D_Re3900_blocks_test_basis',\
                            'DNS300_inc3d_3D_2017_04_02_blocks',\
                            'DNS300_inc3d_3D_2017_04_02_blocks_truncated',\
                            'DNS300_inc3d_3D_2017_04_02_blocks_test_basis',\
                            'test2D_blocks',\
                            'test2D_blocks_truncated',\
                            'test2D_blocks_test_basis',\
                            'small_test_in_blocks',\
                            'small_test_in_blocks_truncated',\
                            'small_test_in_blocks_test_basis',\
                            'DNS100_inc3d_2D_2018_11_16_blocks',\
                            'DNS100_inc3d_2D_2018_11_16_blocks_truncated',\
                            'DNS100_inc3d_2D_2018_11_16_blocks_test_basis',\
                            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks',\
                            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated',\
                            'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_test_basis',\
                            'inc3D_HRLESlong_Re3900_blocks',\
                            'inc3D_HRLESlong_Re3900_blocks_truncated',\
                            'inc3D_HRLESlong_Re3900_blocks_test_basis']
    
            if param['type_data'] in names_list:
                m_U[:,:,0] = m_U[:,:,0] - 1
            
    
            nrj_mU = np.sum(np.power(m_U[:],2))*np.prod(param['dX'])
            param['file_nrj_mU'] = param['folder_results'] + '_nrj_mU_' + param['type_data'] + '.mat'
#            mkdir(param.folder_results)
#            save(param.file_nrj_mU, 'nrj_mU')
        
        
        norm0 = nrj_mU[0,0]
        folder_Path = Path(param['folder_data']).joinpath(param['type_data'] + '_pre_c')
        param['name_file_pre_c'] = str(folder_Path)
        
        param_ref = param
        variables = hdf5storage.loadmat(param['name_file_pre_c'])
        c = variables['c']
        
        
        param = load_param_c(variables['param'])
        
#        load(param.name_file_pre_c,'c','param');
        param_c = param
        param = param_ref
        del param_ref
        c = c*np.prod(param['dX'])/param_c['N_tot'][0,0]
        nrj_varying_tot = np.matrix.trace(c)
#    
#    
    nrj_mean = norm0
    
    del norm0
    
    nrj_tot = nrj_mean + nrj_varying_tot
#    
    if param['reconstruction']:
        err_fix = ((nrj_varying_tot - np.sum(param['lambda'],axis=0)) /nrj_tot) * np.ones(param.N_test,1) 
    else:
        err_fix =  param['truncated_error2']/nrj_tot 
#    
    bt_0 = np.sum(np.power(bt_tot,2),axis=1)[...,np.newaxis]/nrj_tot + err_fix
#    
    bt_forecast_deter = np.sum( np.power(bt_forecast_deter-bt_tot , 2) , axis=1)[...,np.newaxis]/nrj_tot + err_fix
    bt_sans_coef1 =     np.sum( np.power(bt_sans_coef1-bt_tot, 2) , axis=1)[...,np.newaxis]/nrj_tot + err_fix
    bt_forecast_MEV =   np.sum( np.power(bt_forecast_MEV-bt_tot,2),axis=1)[...,np.newaxis]/nrj_tot + err_fix
    struct_bt_MCMC['tot']['mean'] = np.sum(np.power(struct_bt_MCMC['tot']['mean']-bt_tot,2), axis=1)[...,np.newaxis]/nrj_tot + err_fix
    struct_bt_MCMC['tot']['one_realiz'] = np.sum(np.power(struct_bt_MCMC['tot']['one_realiz']-bt_tot,2), axis=1)[...,np.newaxis]/nrj_tot + err_fix
#    
#    
    bt_MCMC = (np.sum(np.power(np.transpose(np.subtract(np.transpose(bt_MCMC,(2,0,1)),bt_tot),(1,2,0)),2),axis=1)/nrj_tot)[:,np.newaxis,:]
    err_tot = np.sum(bt_MCMC,axis = 0)
    
    err_tot_min = np.min(err_tot)
    idx_min_err_tot = np.argmin(err_tot,axis=1)[0]
    
    bt_MCMC_min_error = np.min(bt_MCMC,axis=2)
    idx_min_error = np.argmin(bt_MCMC,axis=2)
    
    
    bt_MCMC_min_error = bt_MCMC_min_error + err_fix
   
    bt_MCMC_RMSE = np.mean(bt_MCMC,axis=2) + err_fix
    del bt_MCMC
    struct_bt_MCMC['tot']['var'] = struct_bt_MCMC['tot']['var']/nrj_tot
    
    
    
    #%%
#    
    
#    if param['nb_modes'] == 2:
#        
#        plt.figure(varying_error_figure)
#        
#        if param['type_data'] == 'turb2D_blocks_truncated':
#            plt.figure(varying_error_figure)
#            plt.axis([X0(1),X0(2),6*width,4*height])
#                
#        elif param['type_data'] == 'incompact3d_wake_episode3_cut':
#            plt.figure(varying_error_figure)
#            plt.axis([X0(1),X0(2),width,height])
#            
#        elif param['type_data'] in ['incompact3d_wake_episode3_cut_truncated','incompact3d_wake_episode3_cut_test_basis']:
#            plt.figure(varying_error_figure)
#            plt.axis([X0(1),X0(2),6*width,4*height])
#        
#        elif param['type_data'] == 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated':
#            plt.figure(varying_error_figure)
#            plt.axis([X0(1), X0(2), 4*width, 4*height])
#        elif param['type_data'] == 'DNS100_inc3d_2D_2018_11_16_blocks_truncated':
#            plt.figure(varying_error_figure)
#            plt.axis([X0(1), X0(2), 4*width, 4*height])
#        elif param['type_data'] in ['LES_3D_tot_sub_sample_blurred','inc3D_Re3900_blocks',\
#                                  'inc3D_Re3900_blocks119']:
#            
#            plt.figure(varying_error_figure)
#            plt.axis([X0(1) ,X0(2) ,6*width, 8*height])
#        elif param['type_data'] =='inc3D_Re3900_blocks_truncated':
#            plt.figure(varying_error_figure)
#            plt.axis([X0(1), X0(2), 4*width, 4*height])
#        else:
#            plt.figure(varying_error_figure)
#            plt.axis([X0(1), X0(2), 4*width, 4*height])
##        
#    else:
    plt.figure(varying_error_figure)
                
#            
#    if (param['type_data'] == 'LES_3D_tot_sub_sample_blurred') or (param['type_data'] == 'inc3D_Re3900_blocks') or \
#        (param['type_data'] == 'inc3D_Re3900_blocks119'):
#            
#        plt.subplot(4,4,np.log2(param.nb_modes))
#    
#    elif (param['type_data'] == 'inc3D_Re3900_blocks_truncated') or (param['type_data'] == 'turb2D_blocks_truncated'):
#        
#        plt.subplot(2,3,np.log2(param.nb_modes))
#        
#    elif param['type_data'] == 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated':
#        plt.subplot(2,2,np.param.nb_modes/2)
#        
#    elif (param['type_data'] == 'DNS100_inc3d_2D_2018_11_16_blocks_truncated') and (param['nb_modes'] <= 16):
#       
#        plt.subplot(2,2,np.log2(param.nb_modes))
#        
#     
#    else:
    #%% Set current subplot
    cols = nb_subplot_cols
    plt.subplot(2,cols,current_subplot)
       
        
    
    
    k=0
    
    # Real values

    
    delta = np.sqrt(np.abs(struct_bt_MCMC['tot']['var'][:,k]))
   
    
    plt.fill_between(time_ref,np.sqrt(err_fix[:,0]),delta+np.sqrt(err_fix[:,0]),color='gray')
#    h = area (time_ref, [np.sqrt(err_fix),delta])
#    
#    set (h(1), 'FaceColor', 'none');
#    set (h(2), 'FaceColor', [0.8 0.8 0.8]);
#    set (h, 'LineStyle', '-', 'LineWidth', 1, 'EdgeColor', 'none');
#    
#   
#    set (gca, 'Layer', 'top');
    
    plt.plot(time,np.sqrt(bt_0),'k')
   
    plt.plot(time,np.sqrt(err_fix),'--k')
    
    if plot_deter:
        plt.plot(time,np.sqrt(bt_forecast_deter[:,k]),'b')
    
    if plot_EV:
        plt.plot(time,np.sqrt(bt_forecast_MEV[:,k]),'b--')
  
    
    plt.plot(time,np.sqrt(struct_bt_MCMC['tot']['mean'][:,k]),'g')
   
    
    plt.plot(time,np.sqrt(bt_MCMC_RMSE[:,k]),'r')
   
    plt.plot(time,np.sqrt(bt_MCMC_min_error[:,k]),'m')
    
    path_save = Path(__file__).parents[3].joinpath('test').joinpath('pythonfile')
    file_to_save = np.vstack((np.sqrt(bt_0)[:,0],np.sqrt(err_fix)[:,0],np.sqrt(bt_forecast_deter[:,k]),np.sqrt(bt_forecast_MEV[:,k]),np.sqrt(struct_bt_MCMC['tot']['mean'][:,k]),\
               np.sqrt(bt_MCMC_RMSE[:,k]),np.sqrt(bt_MCMC_min_error[:,k])))
    
    file_name = str(path_save)
    
    
#    np.save(file_name,file_to_save)
    
    #%%
    
    if np.logical_not(logscale):
        err_min = 0
   
    
    if param['type_data'] == 'incompact3d_wake_episode3_cut':
        err_min = 0.15
            
    elif  param['type_data'] == 'incompact3d_wake_episode3_cut_truncated':
        err_min = 0.005;
        
        if param.nb_modes > 16:
            err_min = 0.005
        
        
        if np.logical_not(logscale):
            err_min = 0
        
            
    elif  param['type_data'] == 'LES_3D_tot_sub_sample_blurred':
        err_min = 0.3
    
    elif  param['type_data'] in [ 'inc3D_Re3900_blocks','inc3D_Re3900_blocks119']:
        err_min = 0.4

    elif  param['type_data'] == 'inc3D_Re3900_blocks_truncated':
        err_min=0.4
    
    else:
        err_min = 0

    
#    ax=[time[0] time[-1] err_min 1 ]
#    axis(ax)
    if logscale:
        pass
#        set(gca,...
#        'YGrid','on', ...
#        'Units','normalized',...
#        'FontUnits','points',...
#        'FontWeight','normal',...
#        'FontSize',FontSize,...
#        'FontName','Times',...
#        'YScale','log')
#        
#        ylabel({'error(log)'},...
#        'FontUnits','points',...
#        'interpreter','latex',...
#        'FontSize',FontSize,...
#        'FontName','Times')
    else:
        plt.ylabel('norm. error')
#        set(gca,...
#            'YGrid','on', ...
#            'Units','normalized',...
#            'FontUnits','points',...
#            'FontWeight','normal',...
#            'FontSize',FontSize,...
#            'FontName','Times')
#        
#        ylabel({'norm. error'},...
#        'FontUnits','points',...
#        'interpreter','latex',...
#        'FontSize',FontSize,...
#        'FontName','Times')
    
#        if param['type_data'] == 'incompact3d_wake_episode3_cut_truncated':
#            ax=[time(1) time(end) err_min 0.37 ];
        
    plt.xlabel('Time')
#    xlabel('Time',...
#    'FontUnits','points',...
#    'FontWeight','normal',...
#    'FontSize',FontSize,...
#    'FontName','Times')
    
    plt.title('n = ' + str(int(param['nb_modes'])))
    plt.grid()
#    title(['$n=' num2str(param.nb_modes) '$'],...
#    'FontUnits','points',...
#    'FontWeight','normal',...
#    'interpreter','latex',...
#    'FontSize',FontSizeTtitle,...
#    'FontName','Times')
#
#    axis(ax)  
#        
    #%% 
    
    
#    if (logscale == True) and (param['type_data'] == 'inc3D_Re3900_blocks_truncated'):
#        err_min=0.45
#        YTick = np.arange(0.4,1+0.1,0.1)
#    
#        set(gca,...
#        'YGrid','on', ...
#        'Units','normalized',...
#        'FontUnits','points',...
#        'FontWeight','normal',...
#        'FontSize',FontSize,...
#        'FontName','Times',...
#        'YTick',YTick,...
#        'YScale','log')
    
#    threshold = str(param['decor_by_subsampl']['spectrum_threshold'])
#    iii = (threshold =='.');
#    threshold(iii)='_';
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    '''
    return idx_min_error, idx_min_err_tot
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    