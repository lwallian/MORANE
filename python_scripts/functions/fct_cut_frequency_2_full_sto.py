# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:22:37 2019

@author: matheus.ladvig
"""
import numpy as np
import math

def fct_cut_frequency_2_full_sto(bt,ILC,param,pchol_cov_noises,modal_dt):
    # Compute how much we can subsample in time the resolved modes with respect to the Shanon
    
    
    
    N_tot, nb_modes = bt.shape
    lambda_fct = param['lambda']
    
    # Normalization
    
    bt =   np.tile(1./np.sqrt(lambda_fct).T,(bt.shape[0],1))*bt

    # If the criterion is on the derivative of the Chronos
#    if param['decor_by_subsampl']['test_fct'] == 'db':
#        # Derivative of Chronos
#        bt = 11111111########
    
    
    spectrum = np.zeros(bt.shape)
    for i in range(nb_modes):
        spec = np.power(np.abs(np.fft.fft(bt[:,i])),2)
        spectrum[:,i] = spec
    
    
    
    
    # Keep the first half of the spectrum (positive frequencies)
    spectrum = spectrum[0:math.ceil(N_tot/2),:]
    freq =   1/N_tot*np.arange(0,math.ceil(N_tot/2))           
    
    # Initialization
    
    spectrum_tot = spectrum
    del spectrum
    
    rate_dt = np.full([nb_modes, 1], np.nan)
    
    for index in range(0,nb_modes):
        
        spectrum = spectrum_tot[:,index]
        spectrum = spectrum[...,np.newaxis]
        #Threshold to determine the numerical zero
        spectrum_threshold = param['decor_by_subsampl']['spectrum_threshold']
        
        max_s = np.max(spectrum)
        threshold = max_s*spectrum_threshold
        
        # Find from which frequency the maximum spectrum is null
        idx_temp = np.where((spectrum > threshold))[0] # give the lines that satisfies the condition
        idx_temp = np.min([idx_temp[-1]+1,len(freq)-1])
        
        freq_cut = freq[idx_temp]
        
    
    
        # Shanon criterion allow the following subsample
        n_subsampl_decor = 1./(2*freq_cut)
        
        #Keep this subsample rate in the interval of possible values
        n_subsampl_decor = np.min([n_subsampl_decor,N_tot])
    
        n_subsampl_decor = np.max([n_subsampl_decor,1])
    
        rate_dt[index] = n_subsampl_decor
    
#    if modal_dt == 2:
#        
#        rate_dt = np.min(rate_dt)*np.ones(rate_dt.shape)
    
    
    print('The time step is modulated by: ' + str(rate_dt.T))
    
    
    #%% Modify Chronos evolution equation
    
    # Finite variation terms
    
    I_deter = ILC['deter']['I']
    L_deter = ILC['deter']['L']
    C_deter = ILC['deter']['C']
    
    I_sto = ILC['sto']['I']
    L_sto = ILC['sto']['L']
    C_sto = ILC['sto']['C']
    
    C_sto = np.multiply(C_sto,rate_dt[:,0,np.newaxis,np.newaxis])
    L_sto = np.multiply(L_sto,rate_dt.T)
    
    for q in range(nb_modes):
        I_sto[q] = -np.trace(np.matmul(np.diag(lambda_fct),C_sto[q,:,:]))
#        I_sto[q] = -np.trace(np.matmul(np.diag(lambda_fct[:,0]),C_sto[q,:,:]))
    
    modal_dt_dict = {'I':I_sto + I_deter,'L':L_sto + L_deter,'C':C_sto + C_deter}
    
    
    ILC['modal_dt'] = modal_dt_dict
    
    
    
    # Martingale terms
    r_rate_dt = np.sqrt(rate_dt)
    weight = np.tile(r_rate_dt.T, (nb_modes, 1))
    weight = np.hstack((r_rate_dt,weight))
    weight = weight.flatten('F')[:,np.newaxis]
    pchol_cov_noises = np.multiply(pchol_cov_noises,weight)
    
    
    
    
    
    return  rate_dt, ILC,pchol_cov_noises














if __name__ == '__main__':
    
    
    pass    