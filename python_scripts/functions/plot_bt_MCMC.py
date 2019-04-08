# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:12:19 2019

@author: matheus.ladvig
"""
import numpy as np
import matplotlib

def plot_bt_MCMC(param,bt_forecast_sto_scalar,bt_forecast_sto_beta,bt_forecast_sto_a_cst_modal_dt, bt_forecast_sto_a_NC_modal_dt, \
                 bt_forecast_deter, bt_forecast_MEV,bt_sans_coef1,bt_sans_coef2,bt_tot,struct_bt_MCMC,plt):
    
    #%% Plot the first coefficients bt along time
    beamer = True
    plot_deter = param['plot']['plot_deter']
    plot_EV = param['plot']['plot_EV']
    plot_tuned = False;
    plot_modal_dt = False;
    
    if beamer:
        
        width = 2.5
        height = 2
    
    else:
        width = 1.5
        height = 1.2
    
    if param['type_data'] == 'incompact3d_wake_episode3_cut_truncated':
        width=2.5
        height=1.5
        
    
    X0 = np.array([0,0])
    
    plot_deter = True

    nb_modes = param['nb_modes']
    
    
    if 'N_tot' in param.keys():
        N_tot = param['N_tot']
        N_test = param['N_test']
        
    struct_bt_MCMC['tot']['mean'] = struct_bt_MCMC['tot']['mean'][0,:int(N_test),:]
    struct_bt_MCMC['tot']['var'] = struct_bt_MCMC['tot']['var'][0,:int(N_test),:]
    struct_bt_MCMC['tot']['one_realiz'] = struct_bt_MCMC['tot']['one_realiz'][:int(N_test),:]
    
    bt_tot = bt_tot[:int(N_test),:]
    bt_forecast_MEV = bt_forecast_MEV[:int(N_test),:]
    bt_forecast_deter = bt_forecast_deter[:int(N_test),:]
    bt_sans_coef1 = bt_sans_coef1[:int(N_test),:]
    struct_bt_MCMC['tot']['one_realiz'] = struct_bt_MCMC['tot']['one_realiz'][:int(N_test),:]
    N_test = N_test-1
    
    dt_tot = param['dt']
    N_time_final = N_tot
    time = np.arange(1,int(N_test+2),1)*dt_tot
    time_ref = time
    
    figures = [manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    number_figures = len(figures) - 1
    for index in range(int(nb_modes)):
        
        plt.figure(number_figures+1+index)
        
    
#        plt.plot(time,struct_bt_MCMC['tot']['one_realiz'][:,index],'y')

        if plot_deter:
            plt.plot(time,bt_forecast_deter[:,index],color = 'b')
        
        
        plt.plot(time,bt_sans_coef1[:,index],'r--')
        
        # Real values
        plt.plot(time_ref,bt_tot[:,index],'k-.')
        
        
#        plt.plot(time_ref,struct_bt_MCMC['tot']['mean'][:,index],'g')
        
        if plot_EV:
            plt.plot(time,bt_forecast_MEV[:,index],'b--')
        
        if plot_tuned:
            plt.plot(time, bt_forecast_sto_scalar[:,index],'c')
            plt.plot(time, bt_forecast_sto_beta[:,index],'y')
    
        if plot_modal_dt:
            plt.plot(time, bt_forecast_sto_a_cst_modal_dt[:,index],'or')
            plt.plot(time, bt_forecast_sto_a_NC_modal_dt[:,index],'om')
            
        plt.grid()
        
        
        plt.ylabel(r'$b_'+str(int(index+1))+'$(t)')
        
#        plt.ylabel(r'$\alpha_i > \beta_i$')
        plt.xlabel('Time')
        plt.title('Temporal mode '+str(int(index)+1))
        
        
        
        
        
    
#    plot(time, (struct_bt_MCMC.tot.one_realiz(:,k))','y');
#    if plot_deter
#        plot(time, (bt_forecast_deter(:,k))','b');
#        %     semilogy(time, (bt_forecast_deter(:,k))','b');
#    end
#    
#    plot(time, (bt_sans_coef1(:,k))','r--');
#    
#    % Real values
#    plot(time_ref,bt_tot(:,k)','k-.');
#    plot(time, (bt_sans_coef1(:,k))','r--');
#    
#    plot(time_ref,struct_bt_MCMC.tot.mean(:,k)','g');
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass