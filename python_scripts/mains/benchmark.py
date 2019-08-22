# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:42:56 2019

@author: matheus.ladvig
"""
from pathlib import Path
import hdf5storage
import sys
path_functions = Path(__file__).parents[1].joinpath('functions')
sys.path.insert(0, str(path_functions))
from evol_forward_bt_RK4 import evol_forward_bt_RK4
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio




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
    
    
    return param_dict.copy(),bt_forecast_MEV,bt_forecast_EV,bt_forecast_NLMEV,variables['ILC_EV']



def benchmark(nb_modes,type_data):


    folder_results = Path(__file__).parents[2].joinpath('resultats').joinpath('current_results')
    
    #### Load matrices from eddy viscosity
    
    file_EV =  'EV_result_' + type_data + '_' + str(nb_modes) + '_modes'
    file_EV = file_EV + '.mat'
    
    folder_results = folder_results.joinpath(file_EV)
    
    param_deter,bt_forecast_MEV,bt_forecast_EV,bt_forecast_NLMEV,ILC_EV = load_file_EV(str(folder_results))
    
    
    I = ILC_EV['EV'][0,0]['I'][0,0]
    L = ILC_EV['EV'][0,0]['L'][0,0]
    C = ILC_EV['EV'][0,0]['C'][0,0]
    
    
                         
    return I,L,C
    

def load_MCMC(nb_modes,beta_1,reynolds):
    path = Path(__file__).parents[3].joinpath('test').joinpath('data_to_benchmark'+ str(nb_modes)+'_bruit_'+str(beta_1)+'reynolds'+str(reynolds))
    variables = hdf5storage.loadmat(str(path))
    
    obs = variables['obs']
    bt_MCMC = variables['bt_MCMC']
    iters = variables['iters']
    dt = variables['dt']
    bt_tot = variables['bt_tot']
    n_simu = variables['n_simu']
    param = variables['param']
    index_of_filtering = variables['index_of_filtering']
#    particles = variables['particles']
    
    
    param_dict = {}
    
    # Loop to run all the dtypes in the data
    for name in param.dtype.names :
        
        if param[0][name][0][0].dtype.names == None:
            param_dict[name] = param[0][name][0]
            
#            if param_deter[0][name].shape == (1,1):
#                param_dict[name] = param_deter[0][name][0,0]
#            else:
#                param_dict[name] = param_deter[0][name]
        # If not, we need to create a dict that will be resposnsible to stock all fields of this structure
        
        else:
            aux_dict = {}
            for name2 in param[0][name][0][0].dtype.names:
                
                    aux_dict[name2] = param[0][name][0][0][name2][0]
                
#            
            param_dict[name] = aux_dict
    
    
    return obs,bt_MCMC,iters[0,0],dt[0,0],bt_tot,n_simu,param_dict.copy(),index_of_filtering#,particles
    

    
    

if __name__ == '__main__':
    
#    type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
    type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
    nb_modes = 6
    beta_1 = 0.1
    reynolds = 100
    
    
    I,L,C = benchmark(nb_modes,type_data)

    obs,bt_MCMC,iters,dt,bt_tot,n_simu,param,index_of_filtering = load_MCMC(nb_modes,beta_1,reynolds)
    
#    steps_to_assimilate = (bt_MCMC.shape[0]-1)/(obs.shape[0]-1)
    bt_forecast_EV = obs[0,:][np.newaxis,...]
    time_assimilate = [0]
    i = 1
    for index in range(iters):
        if index in index_of_filtering:
            bt_evol = obs[i,...][np.newaxis,...]
            print(index)
            time_assimilate.append(index+1)
            i +=1
        else:
            bt_evol = evol_forward_bt_RK4(I,L,C,dt,bt_forecast_EV[-1,...][...,np.newaxis].T)

        bt_forecast_EV = np.concatenate((bt_forecast_EV,bt_evol),axis=0)

    
    
    time_simu = np.arange(0,(bt_MCMC.shape[0])*dt,dt)
    time_bt_tot = np.arange(0,(bt_MCMC.shape[0])*dt,n_simu*dt)
    
    
    for index in range(bt_forecast_EV.shape[1]):
        plt.figure(index)
#        line1 = plt.plot(time_simu,bt_MCMC[:,index],'b-',label = 'Particles mean')
        line2 = plt.plot(time_bt_tot,bt_tot[:,index],'k--',label = 'True state')
        line3 = plt.plot(time_simu,bt_forecast_EV[:,index],'g-',label = 'Eddy viscosity')
        plt.plot(dt*np.array(time_assimilate),-2*np.ones((len(time_assimilate))),'r.')
        plt.grid()

    
    #################################-- Calculate error --######################################
    folder_Path = Path(param['folder_results'][0]).joinpath('_nrj_mU_' + param['type_data'][0] + '0')
    param['file_nrj_mU'] = str(folder_Path)
    type_data_mU = param['type_data'][0] + str(0)
    folder_Path = (Path(param['folder_data'][0]).parents[1]).joinpath('data').joinpath(type_data_mU + '_U_centered')
    param['name_file_mU'] = str(folder_Path)
    variables = hdf5storage.loadmat(str(folder_Path))
    m_U = variables['m_U']
    m_U[:,:,0] = m_U[:,:,0] - 1
    nrj_mU = np.sum(np.power(m_U[:],2))*np.prod(param['dX'])
    param['file_nrj_mU'] = param['folder_results'][0] + '_nrj_mU_' + param['type_data'][0] + '.mat'
    
    norm0 = nrj_mU
    folder_Path = (Path(param['folder_data'][0]).parents[1]).joinpath('data').joinpath(param['type_data'][0] + '_pre_c')
    param['name_file_pre_c'] = str(folder_Path)
    param_ref = param.copy()
    variables = hdf5storage.loadmat(param['name_file_pre_c'])
    c = variables['c']
    
    param = load_param_c(variables['param'])
        
#        load(param.name_file_pre_c,'c','param');
    param_c = param.copy()
    param = param_ref.copy()
    del param_ref
    c = c*np.prod(param['dX'])/param_c['N_tot'][0,0]
    nrj_varying_tot = np.matrix.trace(c)
    
    
    nrj_mean = norm0
    nrj_tot = nrj_mean + nrj_varying_tot
    err_fix =  param['truncated_error2']/nrj_tot
    #########################################################################################################################
    #######################################-- Calculate informations to Plot --###############################################

    bt_0 = np.sum(np.power(bt_tot,2),axis=1)[...,np.newaxis]/nrj_tot + err_fix
    bt_forecast_MEV =  np.sum( np.power(bt_forecast_EV[::int(n_simu)]-bt_tot,2),axis=1)[...,np.newaxis]/nrj_tot + err_fix
    bt_MCMC_assimilated = np.sum( np.power(bt_MCMC[::int(n_simu)]-bt_tot,2),axis=1)[...,np.newaxis]/nrj_tot + err_fix
    
    time = np.arange(0,bt_MCMC_assimilated.shape[0]*dt*n_simu[0,0],dt*n_simu[0,0])
    plt.figure(index+1)
    line1 = plt.plot(time,np.sqrt(bt_0),'k',label = 'No chronos Error')
    line2 = plt.plot(time,np.sqrt(err_fix),'--k',label = 'Trunc. Error')
    line3 = plt.plot(time,np.sqrt(bt_forecast_MEV),'b--',label = 'Eddy Visc.')
    line4 = plt.plot(time,np.sqrt(bt_MCMC_assimilated),'r',label = 'Particle Filtered')
    plt.xlabel('Time(Sec)')
    plt.ylabel('Normalized Error')
    plt.legend()
    plt.grid()


    ######## save data to valentin


    dict_python = {}
    dict_python['obs'] = obs
    dict_python['bt_MCMC'] = bt_MCMC
    dict_python['bt_forecast_EV'] = bt_forecast_EV
    dict_python['iters'] = iters
    dict_python['dt'] = dt
    dict_python['bt_tot'] = bt_tot
    dict_python['n_simu'] = n_simu
    dict_python['param'] = param
    dict_python['index_of_filtering'] = index_of_filtering
    dict_python['beta_noise'] = beta_1
#    dict_python['particles'] = particles
    
    
    
    name_file_data = Path(__file__).parents[3].joinpath('test').joinpath('data_to_valentin_beta_noise_'+str(beta_1)+'Reynolds'+str(reynolds))
    sio.savemat(str(name_file_data),dict_python)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


























