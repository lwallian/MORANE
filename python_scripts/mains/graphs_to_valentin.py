# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:53:26 2019

@author: matheus.ladvig
"""
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_MCMC(nb_modes,beta_1,reynolds):
    path = Path(__file__).parents[3].joinpath('test').joinpath('data_to_benchmark'+ str(nb_modes)+'_bruit_'+str(beta_1))
    variables = hdf5storage.loadmat(str(path))
    
    obs = variables['obs']
    bt_MCMC = variables['bt_MCMC']
    iters = variables['iters']
    dt = variables['dt']
    bt_tot = variables['bt_tot']
    n_simu = variables['n_simu']
    param = variables['param']
    index_of_filtering = variables['index_of_filtering']
    
    
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
    
    
    return obs,bt_MCMC,iters[0,0],dt[0,0],bt_tot,n_simu,param_dict.copy(),index_of_filtering





if __name__ == '__main__':
    type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated'
    #type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'
    nb_modes = 6
    beta_1 = 0.1
    reynolds = 100
    
    
    obs,bt_MCMC,iters,dt,bt_tot,n_simu,param,index_of_filtering = load_MCMC(nb_modes,beta_1,reynolds)
    
    ref = bt_tot
    particles_mean = bt_MCMC
    n_particles = bt_MCMC.shape[-1] 
   
    
    
    quantiles = np.quantile(bt_MCMC[:,:,:],q=[0.025,0.975],axis=2)
      
    
    time_simu = np.arange(0,(bt_MCMC.shape[0])*dt,dt)
    time_bt_tot = np.arange(0,(bt_MCMC.shape[0])*dt,n_simu*dt)
    for index in range(particles_mean.shape[1]):
        plt.figure(index)
        plt.ylim(-10, 10)
        ####        delta = 1.96*particles_std_estimate[:,index]/np.sqrt(n_particles)
        
        plt.fill_between(time_simu,quantiles[0,:,index],quantiles[1,:,index],color='gray')
        line1 = plt.plot(time_simu,particles_mean[:,index],'b-',label = 'Particles mean')
        #        line2 = plt.plot(time_bt_tot,ref[:,index],'k--',label = 'True state')
        line2 = plt.plot(time_bt_tot,ref[:,index],'k--',label = 'True state')
        #        line3 = plt.plot(time_simu,particles_median[:,index],'g.',label = 'particles median')
        #        line4 = plt.plot(dt_tot*np.concatenate((np.zeros((1)),np.array(time_pf))),particles_estimate[:,index],'m.',label = 'PF mean estimation')
        #    plt.plot(dt_tot*np.array(time_pf[1:]),-2*np.ones((len(time_pf[1:]))),'r.')
        plt.grid()
        plt.legend()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    