# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:03:32 2019

@author: matheus.ladvig
"""

import numpy as np
from main_from_existing_ROM_Simulation import main_from_existing_ROM_Simulation
import matplotlib.pyplot as plt
import matplotlib

def super_main_from_existing_ROM_Simulation(vect_nb_modes,type_data,v_threshold,vect_modal_dt,\
                                            no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected):
    
#    plt.figure(0)
    
#    plt.subplot()
    plt.close("all")
    
    nb_modes_max = np.max(vect_nb_modes)
#    nb_modes_total = len(vect_nb_modes)
    vect_nb_modes_len = len(vect_nb_modes)
    if vect_nb_modes_len == 1 or vect_nb_modes_len == 2:
        nb_subplot_cols = 1
    elif vect_nb_modes_len == 3 or vect_nb_modes_len == 4:
        nb_subplot_cols = 2
    else:
        nb_subplot_cols = 3
        
    
    for modal_dt in vect_modal_dt:
        for adv_corrected in vect_adv_corrected:
            for reconstruction in vect_reconstruction:
                for v_thres in v_threshold:
#                    close all
#                    pause(1)
                    figures = [manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
                    number_figures = len(figures)
                    plt.figure(number_figures)
                    varying_error_figure = plt.gcf().number
                    
                    
                        
                    current_subplot = 1
                    for k in vect_nb_modes:
                        
                        main_from_existing_ROM_Simulation(type_data,k, v_thres,no_subampl_in_forecast,\
                                                          reconstruction,adv_corrected,modal_dt,plt,varying_error_figure,nb_subplot_cols,current_subplot)
                        
                        current_subplot = current_subplot + 1
                        
                        
            
                        
if __name__ == '__main__':                       
    vect_nb_modes = [16,8]
    no_subampl_in_forecast = False;
    vect_reconstruction = [False] # for the super_main_from_existing_ROM
    vect_adv_corrected = [False]
    type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated'
    v_threshold = [1e-05]
    vect_modal_dt  = [1e-05]
    super_main_from_existing_ROM_Simulation(vect_nb_modes,type_data,v_threshold,vect_modal_dt,\
                                            no_subampl_in_forecast,vect_reconstruction,vect_adv_corrected)
                        