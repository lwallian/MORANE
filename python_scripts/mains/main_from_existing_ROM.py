# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:17:08 2019

@author: matheus.ladvig
"""
#import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
from scipy import interpolate



#def calculate_sigma_inv(L):
#    
#    sigma = L @ L.T
#    
#    return np.linalg.inv(sigma)


#def calculate_inverse_covariance(L):
#    
#    sigma = np.matmul(L.T,L)
#    
#    sigma_inverse = np.linalg.inv(sigma) 
#    
#    
#    return sigma_inverse

#def filter_space(value,velocity):
#    
#    variance_time = 0.0066
#    constant = -1/(2*variance_time*velocity**2)
#    return np.exp(constant*value**2)

def calculate_H_PIV(topos,new_distance,grid,std_space,only_load,dim,slicing,slice_z):
#    topos ---> 
    '''
    This function applies the spatial filter in the topos.
         - It is necessary to smooth the DNS data and the spatial filter is constructed based
             on estimations on data.
    '''
    
    
    n_modes = topos.shape[1] - 1                                                                       # Number of Chronos solved modes
    path_data = Path(__file__).parents[1].joinpath('data_PIV').joinpath('H_piv_'+str(n_modes)+'.npy')  # Path
    
    
    if only_load==True:             # If the matrix is already calculated, we need only to load it
        matrix = np.load(path_data) # Load 
        return matrix
    
    # If the flow is going in the direction of x
    number_of_points_correlated = int(std_space/(new_distance))                                          # Number of spatial correlated points
    dist = np.abs(new_distance*np.arange(-number_of_points_correlated,number_of_points_correlated+1,1))  # Distances between points   
    h = np.exp(-(dist**2)/(2*std_space**2))                                                              # Constructs the filter
    sum_h = np.sum(h)                                                                                    # Calculates all coefficients
    h = h/sum_h                                                                                          # Normalizes the filter 
    

    
#    h = np.array([0,0,0,0,1,0,0,0,0])
    
    
#    topos = np.transpose(topos,(0,2,1))
#    if dim ==2:
#        matrix_ux = np.zeros(shape=(int(topos.shape[0]/dim),topos.shape[1]))
#        matrix_uy = np.zeros(shape=(int(topos.shape[0]/dim),topos.shape[1]))
#    elif dim==3:
#        matrix_ux = np.zeros(shape=(int(topos.shape[0]/dim),topos.shape[1]))
#        matrix_uy = np.zeros(shape=(int(topos.shape[0]/dim),topos.shape[1]))
#        matrix_uz = np.zeros(shape=(int(topos.shape[0]/dim),topos.shape[1]))
    
    topos = np.reshape(topos,(grid[0],grid[1],grid[2],dim,topos.shape[1]),order='F')  # Reshape the topos
    
    size_x = grid[0]   # Size of x grid
    size_y = grid[1]   # Size of y grid
    size_z = grid[2]   # Size of z grid
    
    
#    i=0
#    for d in range(dim):
#        for z in range(size_z):
##            print(i)
#            for y in range(size_y):
#                
#                a = topos[1:number_of_points_correlated+1,y,z,d,:]
#                b = np.flip(a,axis=0)
#                
#                c = topos[-number_of_points_correlated-1:-1,y,z,d,:]
#                dc = np.flip(c,axis=0)
#                
#                signal = np.concatenate((b,topos[:,y,z,d,:]),axis=0)
#                signal = np.concatenate((signal, dc),axis=0)
#                
#                for j in range(topos.shape[-1]):
#                    matrix[i*size_x:(i+1)*size_x,j] = np.convolve(signal[...,j],h,'valid')
#                    
#                i+=1  
                    
    topos_calcul = np.copy(topos,order='F')  # Define topos_calcul to proceed with the calcul
    
    
    
    
    
    for d in range(dim):                                                       # Define dimension 
        for z in range(size_z):                                                # Define the z slice 
            for y in range(size_y):                                            # Define the line
                a = topos_calcul[1:number_of_points_correlated+1,y,z,d,:]      # Get the border condition 
                b = np.flip(a,axis=0)                                          # Mirrors the condition and it will be the border condition posteriorly
                
                c = topos_calcul[-number_of_points_correlated-1:-1,y,z,d,:]    # Get the border condition at the end
                dc = np.flip(c,axis=0)                                         # Mirrors the condition and it will be the border condition posteriorly
                
                signal = np.concatenate((b,topos_calcul[:,y,z,d,:],dc),axis=0) # Concatentes the data with the border condition
#                signal = np.concatenate((signal, dc),axis=0)
                
                for j in range(topos_calcul.shape[-1]):                        # for all the modes, take the ocnvolution of the signal with the filter 
                    convolution = np.convolve(signal[...,j],h,'valid')         # Convolution and takes only the valid part, ignoring the results with the border condition
                    topos_calcul[:,y,z,d,j] = convolution                      # Stocks it
                    

                
    

    for d in range(dim):                                                       # Define dimension 
        for z in range(size_z):                                                # Define the z slice  
            for x in range(size_x):                                            # Define the line 
                
                a = topos_calcul[x,1:number_of_points_correlated+1,z,d,:]      # Get the border condition 
                b = np.flip(a,axis=0)                                          # Mirrors the condition and it will be the border condition posteriorly
                
                c = topos_calcul[x,-number_of_points_correlated-1:-1,z,d,:]    # Get the border condition at the end
                dc = np.flip(c,axis=0)                                         # Mirrors the condition and it will be the border condition posteriorly
                
                signal = np.concatenate((b,topos_calcul[x,:,z,d,:],dc),axis=0) # Concatentes the data with the border condition 
#                signal = np.concatenate((signal, dc),axis=0)
                
                for j in range(topos_calcul.shape[-1]):                            # for all the modes, take the ocnvolution of the signal with the filter 
                    topos_calcul[x,:,z,d,j] = np.convolve(signal[...,j],h,'valid') # Convolution and takes only the valid part, ignoring the results with the border condition
                    

    
    for d in range(dim):                                                       # Define dimension
        for y in range(size_y):                                                # Define the y slice 
            for x in range(size_x):                                            # Define the line 
                
                a = topos_calcul[x,y,1:number_of_points_correlated+1,d,:]      # Get the border condition
                b = np.flip(a,axis=0)                                          # Mirrors the condition and it will be the border condition posteriorly
                
                c = topos_calcul[x,y,-number_of_points_correlated-1:-1,d,:]    # Get the border condition at the end
                dc = np.flip(c,axis=0)                                         # Mirrors the condition and it will be the border condition posteriorly
                
                signal = np.concatenate((b,topos_calcul[x,y,:,d,:],dc),axis=0) # Concatentes the data with the border condition 
#                signal = np.concatenate((signal, dc),axis=0)
                
                for j in range(topos_calcul.shape[-1]):                             # for all the modes, take the ocnvolution of the signal with the filter 
                    topos_calcul[x,y,:,d,j] = np.convolve(signal[...,j],h,'valid')  # Convolution and takes only the valid part, ignoring the results with the border condition
    
    
    
    
#    i=0
#    for z in range(size_z):
#        for y in range(size_y):
#            
#            a = topos[1:number_of_points_correlated+1,y,z,0,:]
#            b = np.flip(a,axis=0)
#            
#            c = topos[-number_of_points_correlated-1:-1,y,z,0,:]
#            dc = np.flip(c,axis=0)
#            
#            signal = np.concatenate((b,topos[:,y,z,0,:]),axis=0)
#            signal = np.concatenate((signal, dc),axis=0)
#            
#            for j in range(topos.shape[-1]):
#                matrix_ux[i*size_x:(i+1)*size_x,j] = np.convolve(signal[...,j],h,'valid')
#                
#            i+=1
#            
#    i=0
#    for z in range(size_z):
#        for x in range(size_x):
#            
#            a = topos[x,1:number_of_points_correlated+1,z,1,:]
#            b = np.flip(a,axis=0)
#            
#            c = topos[x,-number_of_points_correlated-1:-1,z,1,:]
#            dc = np.flip(c,axis=0)
#            
#            signal = np.concatenate((b,topos[x,:,z,1,:]),axis=0)
#            signal = np.concatenate((signal, dc),axis=0)
#            
#            for j in range(topos.shape[-1]):
#                matrix_uy[i*size_y:(i+1)*size_y,j] = np.convolve(signal[...,j],h,'valid')
#                
#            i+=1
#    if dim==3:      
#        i=0
#        for y in range(size_y):
#            for x in range(size_x):
#                
#                a = topos[x,y,1:number_of_points_correlated+1,2,:]
#                b = np.flip(a,axis=0)
#                
#                c = topos[x,y,-number_of_points_correlated-1:-1,2,:]
#                dc = np.flip(c,axis=0)
#                
#                signal = np.concatenate((b,topos[x,y,:,2,:]),axis=0)
#                signal = np.concatenate((signal, dc),axis=0)
#                
#                for j in range(topos.shape[-1]):
#                    matrix_uz[i*size_z:(i+1)*size_z,j] = np.convolve(signal[...,j],h,'valid')
#                    
#                i+=1



     

#    matrix_ux_res = matrix_ux.reshape((size_x,size_y,size_z,n_modes+1),order='F')[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,:]
##    if slicing == True:
##        matrix_ux_res = matrix_ux_res[:,:,int(slice_z-number_of_points_correlated),:]
#    matrix_ux_res_vector = matrix_ux_res.reshape((int(float(size_y-2*number_of_points_correlated)*(size_x-2*number_of_points_correlated)),n_modes+1),order='F')
#
#
#    matrix_uy_res = matrix_uy.reshape((size_y,size_x,size_z,n_modes+1),order='F')
#    matrix_uy_res_transp = np.transpose(matrix_uy_res,(1,0,2,3))[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,:]
##    if slicing == True:
##        matrix_uy_res_transp = matrix_uy_res_transp[:,:,int(slice_z-number_of_points_correlated),:]
#    matrix_uy_res_transp_vector = matrix_uy_res_transp.reshape((int(float(size_y-2*number_of_points_correlated)*(size_x-2*number_of_points_correlated)),n_modes+1),order='F')
    
#    if dim == 3:
#        matrix_ux_res = matrix_ux.reshape((size_x,size_y,size_z,n_modes+1),order='F')[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,:]
#    #    if slicing == True:
#    #        matrix_ux_res = matrix_ux_res[:,:,int(slice_z-number_of_points_correlated),:]
#        matrix_ux_res_vector = matrix_ux_res.reshape((int(float(size_y-2*number_of_points_correlated)*(size_x-2*number_of_points_correlated)),n_modes+1),order='F')
#    
#    
#        matrix_uy_res = matrix_uy.reshape((size_y,size_x,size_z,n_modes+1),order='F')
#        matrix_uy_res_transp = np.transpose(matrix_uy_res,(1,0,2,3))[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,:]
#    #    if slicing == True:
#    #        matrix_uy_res_transp = matrix_uy_res_transp[:,:,int(slice_z-number_of_points_correlated),:]
#        matrix_uy_res_transp_vector = matrix_uy_res_transp.reshape((int(float(size_y-2*number_of_points_correlated)*(size_x-2*number_of_points_correlated)),n_modes+1),order='F')
#
#        matrix_uz_res = matrix_uz.reshape((size_z,size_x,size_y,n_modes+1),order='F')
#        matrix_uz_res_transp = np.transpose(matrix_uz_res,(1,2,0,3))[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,:]
##        if slicing == True:
##            matrix_uz_res_transp = matrix_uz_res_transp[:,:,int(slice_z-number_of_points_correlated),:]
#        matrix_uz_res_transp_vector = matrix_uz_res_transp.reshape((int((size_y-2*number_of_points_correlated)*float(size_x-2*number_of_points_correlated)),n_modes+1),order='F')
#        matrix = np.concatenate((matrix_ux_res_vector,matrix_uy_res_transp_vector,matrix_uz_res_transp_vector))
#
#    elif dim==2:
#        
#        matrix_ux_res = matrix_ux.reshape((size_x,size_y,size_z,n_modes+1),order='F')[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,...]
#        matrix_ux_res_vector = matrix_ux_res.reshape((int(float(size_y-2*number_of_points_correlated)*(size_x-2*number_of_points_correlated)),n_modes+1),order='F')
#        
#        matrix_uy_res = matrix_uy.reshape((size_y,size_x,size_z,n_modes+1),order='F')
#        matrix_uy_res_transp = np.transpose(matrix_uy_res,(1,0,2,3))[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,...]
#        matrix_uy_res_transp_vector = matrix_uy_res_transp.reshape((int(float(size_y-2*number_of_points_correlated)*(size_x-2*number_of_points_correlated)),n_modes+1),order='F')
#        
#        
#        matrix = np.concatenate((matrix_ux_res_vector,matrix_uy_res_transp_vector))

    





                   
#                matrix[i*size_x:(i+1)*size_x,0] = np.convolve(signal[...,0],h/sum_h,'valid')
#                matrix[i*size_x:(i+1)*size_x,1] = np.convolve(signal[...,1],h/sum_h,'valid')
#                matrix[i*size_x:(i+1)*size_x,2] = np.convolve(signal[...,2],h/sum_h,'valid')
#                matrix[i*size_x:(i+1)*size_x,3] = np.convolve(signal[...,3],h/sum_h,'valid')
#                matrix[i*size_x:(i+1)*size_x,4] = np.convolve(signal[...,4],h/sum_h,'valid')
#                matrix[i*size_x:(i+1)*size_x,5] = np.convolve(signal[...,5],h/sum_h,'valid')
#                matrix[i*size_x:(i+1)*size_x,6] = np.convolve(signal[...,6],h/sum_h,'valid')
#                i+=1
                
                
#                for x in range(size_x):
#                    
#                    
#                    
#                    
#                    
#                    if x==0:
#                        matrix[i,0] = h[0]*topos[x,y,z,d,0] + 2*h[1]*topos[x+1,y,z,d,0]
#                        matrix[i,1] = h[0]*topos[x,y,z,d,1] + 2*h[1]*topos[x+1,y,z,d,1]
#                        matrix[i,2] = h[0]*topos[x,y,z,d,2] + 2*h[1]*topos[x+1,y,z,d,2]
#                        matrix[i,3] = h[0]*topos[x,y,z,d,3] + 2*h[1]*topos[x+1,y,z,d,3]
#                        matrix[i,4] = h[0]*topos[x,y,z,d,4] + 2*h[1]*topos[x+1,y,z,d,4]
#                        matrix[i,5] = h[0]*topos[x,y,z,d,5] + 2*h[1]*topos[x+1,y,z,d,5]
#                        matrix[i,6] = h[0]*topos[x,y,z,d,6] + 2*h[1]*topos[x+1,y,z,d,6]
#                        
#                        
#                    elif x==(size_x-1):
#                        
#                        matrix[i,0] = 2*h[1]*topos[x-1,y,z,d,0] + h[0]*topos[x,y,z,d,0]
#                        matrix[i,1] = 2*h[1]*topos[x-1,y,z,d,1] + h[0]*topos[x,y,z,d,1]
#                        matrix[i,2] = 2*h[1]*topos[x-1,y,z,d,2] + h[0]*topos[x,y,z,d,2] 
#                        matrix[i,3] = 2*h[1]*topos[x-1,y,z,d,3] + h[0]*topos[x,y,z,d,3] 
#                        matrix[i,4] = 2*h[1]*topos[x-1,y,z,d,4] + h[0]*topos[x,y,z,d,4] 
#                        matrix[i,5] = 2*h[1]*topos[x-1,y,z,d,5] + h[0]*topos[x,y,z,d,5] 
#                        matrix[i,6] = 2*h[1]*topos[x-1,y,z,d,6] + h[0]*topos[x,y,z,d,6] 
#                    
#                    else:
#                        matrix[i,0] = h[1]*topos[x-1,y,z,d,0] + h[0]*topos[x,y,z,d,0] + h[1]*topos[x+1,y,z,d,0]
#                        matrix[i,1] = h[1]*topos[x-1,y,z,d,1] + h[0]*topos[x,y,z,d,1] + h[1]*topos[x+1,y,z,d,1]
#                        matrix[i,2] = h[1]*topos[x-1,y,z,d,2] + h[0]*topos[x,y,z,d,2] + h[1]*topos[x+1,y,z,d,2]
#                        matrix[i,3] = h[1]*topos[x-1,y,z,d,3] + h[0]*topos[x,y,z,d,3] + h[1]*topos[x+1,y,z,d,3]
#                        matrix[i,4] = h[1]*topos[x-1,y,z,d,4] + h[0]*topos[x,y,z,d,4] + h[1]*topos[x+1,y,z,d,4]
#                        matrix[i,5] = h[1]*topos[x-1,y,z,d,5] + h[0]*topos[x,y,z,d,5] + h[1]*topos[x+1,y,z,d,5]
#                        matrix[i,6] = h[1]*topos[x-1,y,z,d,6] + h[0]*topos[x,y,z,d,6] + h[1]*topos[x+1,y,z,d,6]
                    
                    
                    
#                    i+=1
    
    
    
    
    
#    topos = np.reshape(topos,(topos.shape[0]*topos.shape[1],topos.shape[2]),order='F')
#    
#    matrix = 50*np.zeros(shape=(topos.shape[0],7))
#    
#
#    
#    h = [0.6,0.2]
#    
#    
#    len_z = grid[0,2]
#    len_y = grid[0,1]
#    len_x = grid[0,0]
#    
#    i = 0
#    for d in range(dim):
#        for z in range(len_z):
#            print(i)
#            for y in range(len_y):
#                for x in range(len_x):
#                    
#                    index = int((d)*(len_z)*(len_y)*(len_x) + z*(len_y)*(len_x) + y*(len_x) + x)
#                    if x==0:
#                        matrix[i,0] = h[0]*topos[index,0] + h[1]*topos[int(index+1),0]
#                        matrix[i,1] = h[0]*topos[index,1] + h[1]*topos[int(index+1),1]
#                        matrix[i,2] = h[0]*topos[index,2] + h[1]*topos[int(index+1),2]
#                        matrix[i,3] = h[0]*topos[index,3] + h[1]*topos[int(index+1),3]
#                        matrix[i,4] = h[0]*topos[index,4] + h[1]*topos[int(index+1),4]
#                        matrix[i,5] = h[0]*topos[index,5] + h[1]*topos[int(index+1),5]
#                        matrix[i,6] = h[0]*topos[index,6] + h[1]*topos[int(index+1),6]
#                        
#                    elif x == (len_x-1):
#                        
#                        matrix[i,0] = h[1]*topos[int(index-1),0] + h[0]*topos[index,0]
#                        matrix[i,1] = h[1]*topos[int(index-1),1] + h[0]*topos[index,1]
#                        matrix[i,2] = h[1]*topos[int(index-1),2] + h[0]*topos[index,2]
#                        matrix[i,3] = h[1]*topos[int(index-1),3] + h[0]*topos[index,3]
#                        matrix[i,4] = h[1]*topos[int(index-1),4] + h[0]*topos[index,4]
#                        matrix[i,5] = h[1]*topos[int(index-1),5] + h[0]*topos[index,5]
#                        matrix[i,6] = h[1]*topos[int(index-1),6] + h[0]*topos[index,6]
#            
#                    else:
#                        
#                        matrix[i,0] = h[1]*topos[int(index-1),0] + h[0]*topos[index,0] + h[1]*topos[index+1,0]
#                        matrix[i,1] = h[1]*topos[int(index-1),1] + h[0]*topos[index,1] + h[1]*topos[index+1,1]
#                        matrix[i,2] = h[1]*topos[int(index-1),2] + h[0]*topos[index,2] + h[1]*topos[index+1,2]
#                        matrix[i,3] = h[1]*topos[int(index-1),3] + h[0]*topos[index,3] + h[1]*topos[index+1,3]
#                        matrix[i,4] = h[1]*topos[int(index-1),4] + h[0]*topos[index,4] + h[1]*topos[index+1,4]
#                        matrix[i,5] = h[1]*topos[int(index-1),5] + h[0]*topos[index,5] + h[1]*topos[index+1,5]
#                        matrix[i,6] = h[1]*topos[int(index-1),6] + h[0]*topos[index,6] + h[1]*topos[index+1,6]
#        
#                    
#                    i+=1
                    
                    
#    np.save(path_data,matrix)               
    
    return topos_calcul,number_of_points_correlated
    
#def calculate_residue_and_return_reconstructed_flow(topos_l,champ,bt_tot):
#    champ_a =  champ
#    bt_tot = bt_tot[:int((champ.shape[3]-1)/3)+1,:]
#    bt_tot = np.concatenate((bt_tot,np.ones((bt_tot.shape[0],1))),axis=1).T
#    x,y,z,t,dim = champ.shape
#    champ = np.transpose(champ,(0,1,2,4,3))
#    champ = np.reshape(champ,(champ.shape[0]*champ.shape[1]*champ.shape[2]*champ.shape[3],champ.shape[4]),order='F')
##    a = champ[:,::3]       
#    champ = champ[:,::3]
#        
#    champ_reconstructed =  topos_l @ bt_tot   
#
#    value_abs_residue = np.abs(np.subtract(champ,champ_reconstructed))
#                         
#    print('Max grid pointwise difference in flow reconstructed to flow DNS: '+str(np.max(value_abs_residue)))
#    print('Min grid pointwise difference in flow reconstructed to flow DNS: '+str(np.min(value_abs_residue)))
#    print('Average grid pointwise difference in flow reconstructed to flow DNS for all time reconstruction: '+str(np.mean(value_abs_residue,axis=0)))
#    champ_reconstructed = champ_reconstructed.reshape((x,y,z,dim,champ_reconstructed.shape[1]),order='F')
#    
#    
##    erreur = np.subtract(champ_a[:,:,:,0,:].ravel(),champ_reconstructed[:,:,:,:,0].ravel())
#    
#    return champ_reconstructed
#    
    
#def load_cov_tensor(path):
#    
#    tensor_var = hdf5storage.loadmat(str(path))
#    
#    return tensor_var['z'][:,0,:,:]
#
#def calculate_noise_covariance_tensor(tensor_var,L,grid,MX,std_space,subsampling_grid,dim,dt_optimal,slicing,slice_z):
#    
#    tensor_var = tensor_var/dt_optimal
#    
#    
#    variance_uxx = tensor_var[:,0,0]
#    variance_uxy = tensor_var[:,0,1]
#    variance_uxz = tensor_var[:,0,2]
#    
#    variance_uyx = tensor_var[:,1,0]
#    variance_uyy = tensor_var[:,1,1]
#    variance_uyz = tensor_var[:,1,2]
#    
#    variance_uzx = tensor_var[:,2,0]
#    variance_uzy = tensor_var[:,2,1]
#    variance_uzz = tensor_var[:,2,2]
#    
#    
#    number_of_points_correlated = int(std_space/(grid[0,0]))
#    dist = np.abs(grid[0,0]*np.arange(-number_of_points_correlated,number_of_points_correlated+1,1))
#    h = np.exp(-(dist**2)/(2*std_space**2))
#    h = h/np.sum(h)
#    
##    h = np.array([0,0,0,0,1,0,0,0,0])
#    h = h**2
#    
#    dim_x,dim_y,dim_z = MX[0]
#    
#    variance_uxx = variance_uxx.reshape((dim_x,dim_y,dim_z),order='F')
#    variance_uxy = variance_uxy.reshape((dim_x,dim_y,dim_z),order='F')
#    variance_uxz = variance_uxz.reshape((dim_x,dim_y,dim_z),order='F')
#    
#    variance_uyx = variance_uyx.reshape((dim_x,dim_y,dim_z),order='F')
#    variance_uyy = variance_uyy.reshape((dim_x,dim_y,dim_z),order='F')
#    variance_uyz = variance_uyz.reshape((dim_x,dim_y,dim_z),order='F')
#    
#    variance_uzx = variance_uzx.reshape((dim_x,dim_y,dim_z),order='F')
#    variance_uzy = variance_uzy.reshape((dim_x,dim_y,dim_z),order='F')
#    variance_uzz = variance_uzz.reshape((dim_x,dim_y,dim_z),order='F')
#    
#    
#    variances_ux = {}
#    variances_ux['variance_uxx'] = variance_uxx
#    variances_ux['variance_uxy'] = variance_uxy
#    variances_ux['variance_uxz'] = variance_uxz
#    
#    variances_uy = {}
#    variances_uy['variance_uyx'] = variance_uyx
#    variances_uy['variance_uyy'] = variance_uyy
#    variances_uy['variance_uyz'] = variance_uyz
#    
#    variances_uz = {}
#    variances_uz['variance_uzx'] = variance_uzx
#    variances_uz['variance_uzy'] = variance_uzy
#    variances_uz['variance_uzz'] = variance_uzz
#    
#    
#    
#    
##    h_vector = np.tile(h,dim_x)
#    matrix_ux = np.zeros(shape=(dim_x,dim_y,dim_z,int(len(variances_ux.keys()))))
#    j=0
#    for cross,cov_relation in enumerate(variances_ux.keys()):
#        
#        for z in range(dim_z):
#            for y in range(dim_y):
#                vector = np.convolve(np.concatenate((variances_ux[cov_relation][1:number_of_points_correlated+1,y,z][::-1],variances_ux[cov_relation][:,y,z],variances_ux[cov_relation][-number_of_points_correlated-1:-1,y,z][::-1])),h,'valid')
#                matrix_ux[:,y,z,cross] = vector
#                j+=1
#    
#    
#    
#    
#    
#    matrix_uy = np.zeros(shape=(dim_x,dim_y,dim_z,int(len(variances_ux.keys()))))
#    j=0
#    for cross,cov_relation in enumerate(variances_uy.keys()):
#        for z in range(dim_z):
#            for x in range(dim_x):
#                vector = np.convolve(np.concatenate((variances_uy[cov_relation][x,1:number_of_points_correlated+1,z][::-1],variances_uy[cov_relation][x,:,z],variances_uy[cov_relation][x,-number_of_points_correlated-1:-1,z][::-1])),h,'valid')
#                matrix_uy[x,:,z,cross] = vector
#                j+=1
#    
#    matrix_uz = np.zeros(shape=(dim_x,dim_y,dim_z,int(len(variances_ux.keys()))))
#    j=0
#    for cross,cov_relation in enumerate(variances_uz.keys()):
#        for y in range(dim_y):
#            for x in range(dim_x):
#                vector = np.convolve(np.concatenate((variances_uz[cov_relation][x,y,1:number_of_points_correlated+1][::-1],variances_uz[cov_relation][x,y,:],variances_uz[cov_relation][x,y,-number_of_points_correlated-1:-1][::-1])),h,'valid')
#                matrix_uz[x,y,:,cross] = vector
#                j+=1
#    
#    
#    ################## send to trash values with bondary influences 
#    matrix_ux = matrix_ux[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,:]
#    matrix_uy = matrix_uy[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,:]
#    matrix_uz = matrix_uz[number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,number_of_points_correlated:-number_of_points_correlated,:]
#    
#    
#    if slicing ==True:
#        matrix_ux = matrix_ux[:,:,int(slice_z-number_of_points_correlated):int(slice_z-number_of_points_correlated+1),:]
#        matrix_uy = matrix_uy[:,:,int(slice_z-number_of_points_correlated):int(slice_z-number_of_points_correlated+1),:]
#        matrix_uz = matrix_uz[:,:,int(slice_z-number_of_points_correlated):int(slice_z-number_of_points_correlated+1),:]
#    
#    new_dim_x,new_dim_y,new_dim_z,cross = matrix_ux.shape
#    matrices_to_inverse = np.zeros(shape=(int(new_dim_x*new_dim_y*new_dim_z),cross,cross))
#    n=0
#    for z in range(new_dim_z):
#        for y in range(new_dim_y):
#            for x in range(new_dim_x):
#                matrices_to_inverse[n,:,:] = np.concatenate((matrix_ux[x,y,z,:][np.newaxis,...],matrix_uy[x,y,z,:][np.newaxis,...],matrix_uz[x,y,z,:][np.newaxis,...]),axis=0) + np.diag((L**2)*np.ones(cross))
#                n+=1
#    
#    matrix_inversed = np.zeros(shape=(matrices_to_inverse.shape))
#    for i in range(matrices_to_inverse.shape[0]):
#        matrix_inversed[i,:,:] = np.linalg.inv(matrices_to_inverse[i,:,:])
#    
#    vector = np.zeros((int(matrix_inversed.shape[0]*matrix_inversed.shape[1])))
#    n=0
#    for i in range(matrix_inversed.shape[1]):
#        for k in range(matrix_inversed.shape[0]):
#            vector[n] = matrix_inversed[k,i,i]
#            n+=1
    
#    multiplication_matrix = np.zeros(shape=(int(dim_y*dim_z*len(variances.keys())),h_vector.shape[0])) matrix_uz[:,:,0,:]
#    j=0
#    for cov_relation in variances.keys():
#        for z in range(dim_z):
#            for y in range(dim_y):
#                nb_zeros = np.abs(0-number_of_points_correlated)
#                value = np.concatenate((np.zeros(nb_zeros),variances[cov_relation][0,y,z][np.newaxis],2*variances[cov_relation][1:nb_zeros+1,y,z]))
#                vector = value 
#                for i in range(1,dim_x):
#                    
#                    if i<number_of_points_correlated:
#                        nb_zeros = np.abs(i-number_of_points_correlated)
#                        value = np.concatenate((np.zeros(nb_zeros),variances[cov_relation][0:int(2*i+1),y,z],2*variances[cov_relation][int(2*i+1):int(i+number_of_points_correlated+1),y,z]))
#                        vector = np.concatenate((vector,value))
#                    
#                    elif (i+1+number_of_points_correlated)>dim_x:
#                        nb_zeros = np.abs(i+1+number_of_points_correlated-dim_x)
#                        value = np.concatenate((2*variances[cov_relation][i-number_of_points_correlated:i-number_of_points_correlated+nb_zeros,y,z],variances[cov_relation][i-number_of_points_correlated+nb_zeros:i+(number_of_points_correlated-nb_zeros)+1,y,z],np.zeros(nb_zeros)))
#                        vector = np.concatenate((vector,value))
#                        
#                        
#                        
#                    else:
#                        vector = np.concatenate((vector,variances[cov_relation][i-number_of_points_correlated:i+number_of_points_correlated+1,y,z]))
#                    
#                    
#                results = np.multiply(vector,h_vector)
#                multiplication_matrix[j,:] = results
#                    
#                j+=1
#        
#        
#            
#    ########## If diagonal
#    multiplication_matrix = multiplication_matrix[:int(3*dim_y*dim_z),:]
#    
#    multiplication_matrix_H = np.multiply(multiplication_matrix,np.tile(h_vector,(multiplication_matrix.shape[0],1)))
#    
#    multiplication_matrix_H_reshaped = multiplication_matrix_H.reshape((dim_y*dim_z*dim_x*3,int(2*number_of_points_correlated+1)),order='C')
#    
#    multiplication_matrix_H = np.sum(multiplication_matrix_H_reshaped,axis=1)
#    
#    
##    variance_ux = tensor_var[:,0,0]
##    variance_uy = tensor_var[:,1,1]
##    variance_uz = tensor_var[:,2,2]
##    sigma = np.concatenate((variance_ux,variance_uy,variance_uz))
##    Hpiv_sigma = calculate_H_PIV(np.sqrt(sigma),dX,grid,std_space,False,subsampling_grid,dim)
#    
#    Sigma_not_resolved  = L**2 + multiplication_matrix_H
##    
##    
#    Sigma_inversed_diagonal = (Sigma_not_resolved**(-1))
    
    
#    return vector


#def reduce_and_interpolate_topos(topos_dns,grid_dns,MX,dim,slicing,slice_z):
#    x0 = 75.60
#    y0 = 0.75
#    
#    x0_dns = 60
#    y0_dns = 49
#    
#    
#    topos_dns = np.reshape(topos_dns,newshape =(MX[0],MX[1],MX[2],topos_dns.shape[-2],dim),order='F')
#    
#    if slicing == True:
#        
#        topos_dns = topos_dns[:,:,slice_z,...]
#        
#    
#    '''
#    First, we must load one part of the PIV configuration to analise the PIV grid to posterior processing of data PIV and DNS. The data must be measured in
#    the same points.
#    '''
#    file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('wake_Re300_export_190709_0100').joinpath('B0001.dat')
#    data = open(str(file))
#    datContent = [i.strip().split() for i in data.readlines()]
#    data = datContent[4:]
#    
#    nb_lines = len(data)
#    nb_collums = len(data[0])
#    
#    matrix = np.zeros(shape=(nb_lines,nb_collums))
#    
#    for i,line in enumerate(data):
#        for j,number in enumerate(line):
#            matrix[i,j] = number
#    
#    ##################################################### speed in m/s
#        
#        
#    #        |x(mm)|y(mm)|vx(m/s)|vy(m/s)|isValid|
#            
#    '''
#    
#    - Normalise velocity per infinity velocity
#    - Normalise grid per cilinder diameter
#    
#    '''
#    
#    u_inf_measured = 0.388 # m/s
#    cil_diameter = 12 # 12mm
#    
#    
#    grid = matrix[:,0:2] 
#    valid = matrix[:,4] # Get the effective values
#    matrix_valid_grid = matrix[np.where((valid==1))[0],:] # Select only the points that have a value
#    matrix_valid_grid[:,2:4] = matrix_valid_grid[:,2:4]/u_inf_measured
#    
#    
#    ############### PIV centering and unity of measure to 1/D  #########################
#
#    '''
#    This PIV data coordiantes are ((xo ; yo) = (-75.60 ; 0.75) )
#    '''
#    
#    
#    matrix_valid_grid_new_coord_x = (matrix_valid_grid[:,0] + x0)/cil_diameter
#    matrix_valid_grid_new_coord_y = (matrix_valid_grid[:,1] - y0)/cil_diameter
#    matrix_valid_grid[:,0] = (matrix_valid_grid[:,0] + x0)/cil_diameter
#    matrix_valid_grid[:,1] = (matrix_valid_grid[:,1] - y0)/cil_diameter
#    
#    
#    '''
#    - Matrix_valid_grid_new_coord_x is the x distance vector from the points to the center of the cylinder
#    - Matrix_valid_grid_new_coord_y is the y distance vector from the points to the center of the cylinder
#    
#    
#    '''
#    
#    ##################################### DNS centering ####################################
#    
#    '''
#    - The center of the cilynder is the position 60 in the vector x and 49 in vector y and we must centralise the coordinates beggining in these points 
#    
#    '''
#    
#    
#    grid_dns_x = grid_dns[0]
#    grid_dns_y = grid_dns[1]
#    
#    
#    grid_dns_x_centralised = grid_dns_x[:,0] - grid_dns_x[x0_dns,0]
#    grid_dns_y_centralised = grid_dns_y[:,0] - grid_dns_y[y0_dns,0]
#
#    
#    ################################--Cutting less accurate pixels of PIV--#######################################################
#
#    
#    '''
#    The grid is variable because the algorithm of PIV dont give us a square grid. Then We must find the point to cut and transform it in a 
#    squared grid to posterior 2d interpolations 
#    
#    '''
#    
#    end_points = []
#    start_points = []
#    
#    
#    start_points.append([0,matrix_valid_grid_new_coord_x[0]])
#    for i,value in enumerate(matrix_valid_grid_new_coord_x[:-1]):
#        if (matrix_valid_grid_new_coord_x[i+1]-value<0):
#            end_points.append([i,value])
#            start_points.append([i+1,matrix_valid_grid_new_coord_x[i+1]])
#    
#    end_points.append([i+1,matrix_valid_grid_new_coord_x[i+1]])
#    
#    
#    '''
#    - The first line of x and the last one seems to be always corrupted, so i'll cut off the first and the last x grid line
#    
#    - The vector matrix_valid_grid_new_coord_x will begin in 34(The first 34 are samples are incomplete)
#    - The vector will end in 28171(The last line of x is incomplete).
#    '''
#    
#    matrix_valid_grid = matrix_valid_grid[start_points[1][0]:end_points[-2][0]+1]
#    
#    matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[start_points[1][0]:end_points[-2][0]+1]
#    matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[start_points[1][0]:end_points[-2][0]+1]
#    start_points = start_points[1:-1]
#    end_points = end_points[1:-1]
#    
#    '''
#    Now it's necessary to find the squared grid 
#    
#    '''
#    start_points = np.array(start_points)
#    end_points = np.array(end_points)
#    
#    
#    start_x = np.max(start_points[:,1])
#    end_x = np.min(end_points[:,1])
#    
#    
#    indexes_of_grid = np.where((matrix_valid_grid_new_coord_x>=start_x)&(matrix_valid_grid_new_coord_x<=end_x))
#    
#    
#    
#    '''
#    Selecting only the elements inside the grid with limits [start_x, end_x] --> the results is a squared grid 
#    '''
#    matrix_valid_grid = matrix_valid_grid[indexes_of_grid]
#    
#    matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[indexes_of_grid]
#    matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[indexes_of_grid]
#
#
#    ########################### 2nd part ########################
#    '''
#    Cut 'n'(3,4,5...????) pixels from the PIV squared grid ---> Estimation algorithm less accurate in the window boundary 
#    
#    '''
#    
#    n = 3
#    
#    value_in_n_start = matrix_valid_grid_new_coord_x[n]
#    value_in_n_end = matrix_valid_grid_new_coord_x[-n-1]
#    
#    
#    indexes_of_grid_n = np.where((matrix_valid_grid_new_coord_x>=value_in_n_start)&(matrix_valid_grid_new_coord_x<=value_in_n_end))
#    matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[indexes_of_grid_n]
#    matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[indexes_of_grid_n]
#    matrix_valid_grid = matrix_valid_grid[indexes_of_grid_n]
#    
#    
#    ######################### 3rd part  ####################
#    '''
#    The cilynder is not centralised in the window,(i.e the number of sampled lines in left is different from the right of the cilynder), we must centralise it, selecting 
#    the same number in both sides and a sample to be the centered(The nearest sample to 0). 
#    '''
#    
#    sampled_points_in_y = np.unique(matrix_valid_grid_new_coord_y)
#    indexes_pos = np.where((sampled_points_in_y>0))[0]
#    indexes_neg = np.where((sampled_points_in_y<0))[0]
#    positive = len(indexes_pos)
#    negative = len(indexes_neg)
#    
#    if positive>negative:
#        indexes_pos = indexes_pos[:negative]
#        sampled_points_in_y = sampled_points_in_y[np.concatenate((indexes_neg,indexes_pos))]
#       
#        if sampled_points_in_y[np.argmin(np.abs(sampled_points_in_y))]<0:
#            sampled_points_in_y = sampled_points_in_y[:-1]
#        elif sampled_points_in_y[np.argmin(np.abs(sampled_points_in_y))]>0:
#            sampled_points_in_y = sampled_points_in_y[1:]
#        
#        indexes_after_find_equal_sides = np.where((matrix_valid_grid_new_coord_y<=sampled_points_in_y[-1]))
#        matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[indexes_after_find_equal_sides]
#        matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[indexes_after_find_equal_sides]
#        matrix_valid_grid = matrix_valid_grid[indexes_after_find_equal_sides]
#    elif positive<negative:
#        indexes_neg = indexes_neg[:positive]
#        sampled_points_in_y = sampled_points_in_y[np.concatenate((indexes_neg,indexes_pos))]
#        if sampled_points_in_y[np.argmin(np.abs(sampled_points_in_y))]<0:
#            sampled_points_in_y = sampled_points_in_y[:-1]
#        elif sampled_points_in_y[np.argmin(np.abs(sampled_points_in_y))]>0:
#            sampled_points_in_y = sampled_points_in_y[1:]
#        
#        
#        indexes_after_find_equal_sides = np.where((matrix_valid_grid_new_coord_y>=sampled_points_in_y[0]))
#        matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[indexes_after_find_equal_sides]
#        matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[indexes_after_find_equal_sides]
#        matrix_valid_grid = matrix_valid_grid[indexes_after_find_equal_sides]
#    
#    '''
#    In this line, the vector 'sampled_points_in_y' represents the sampled points in y with the same quantity in both sides 
#    
#    '''
#
#    
#    ##################### 4th part  ###################
#
#    '''
#    Here, we begin to process the DNS grid, first we need to centralise the cilynder in the middle of the window DNS grid as we did with PIV grid
#    
#    ''' 
#    
#    indexes_pos = np.where((grid_dns_y_centralised>0))[0]
#    indexes_neg = np.where((grid_dns_y_centralised<0))[0]
#    positive = len(indexes_pos)
#    negative = len(indexes_neg)
#    
#    if positive>negative:
#        indexes_pos = indexes_pos[:negative]
#    
#    elif positive<negative:
#        indexes_neg = indexes_neg[int(negative-positive):]
#    
#    
#    grid_dns_y_centralised = grid_dns_y_centralised[np.concatenate((indexes_neg,np.array([indexes_neg[-1]+1]),indexes_pos))]
#    topos_dns = topos_dns[:,np.concatenate((indexes_neg,np.array([indexes_neg[-1]+1]),indexes_pos)),:,:]
#    
#    
#    
#    
#    ################### 5 th part 
#    x_piv = np.unique(matrix_valid_grid_new_coord_x)
#    y_piv = np.unique(matrix_valid_grid_new_coord_y)
#    
#    
#    '''
#    The dimensions of PIV and DNS are not the same, therefore we must find the same number of sampled points for each dimension
#    PIV_range ---->  x = (0.74,10.44) y=(-2.84,2.83)
#    DNS_range ---->  x = (-2.5,15.04) y=(-1.95,1.95)
#    
#    Therefore, the effective window(where we have information about model and obs) is the intersection between both ones.  
#    
#            
#                    
#                    ----------------------------       -   -    -    -
#                    |                          |                     |
#        ------------------------------------------------    -
#        |        ---                           |       |    |        |
#        |       -----                          |       |   3.9      5.68
#        |        ---                           |       |    |        |
#        ------------------------------------------------    -           ---------------->DNS window
#                    |                          |                     |
#                    ----------------------------      -    -    -    -  ---------------->PIV window
#        
#                    |<    -  -   11.18   -  - >|         
#        
#        
#        |<     -       -         17.19      -     -   >|
#    
#    
#    '''
#    
#    
#    first_x_PIV = x_piv[0]
#    last_x_PIV = x_piv[-1]
#    index_min = np.where((grid_dns_x_centralised>first_x_PIV))[0][0]
#    index_max = np.where((grid_dns_x_centralised>last_x_PIV))[0][0]
#    
#    
#    grid_dns_x_centralised = grid_dns_x_centralised[index_min-1:index_max+1]
#    topos_dns = topos_dns[index_min-1:index_max+1,...]
#    
#        
#    
#    ##########
#
#    extreme_left_DNS = grid_dns_y_centralised[0]
#    extreme_right_DNS = grid_dns_y_centralised[-1]
#    
#    index_min = np.where((y_piv<extreme_left_DNS))[0][-1]
#    index_max = np.where((y_piv>extreme_right_DNS))[0][0]
#    
#    
#    y_piv = y_piv[index_min+1:index_max]
#    
#    
#    '''
#    Now selecting the points in matrix valid grid that the collum of y is the same of y_piv
#    '''
#    
#    indexes_matrix_valid_grid = np.where((matrix_valid_grid[:,1]>=y_piv[0])&(matrix_valid_grid[:,1]<=y_piv[-1]))[0]
#    
#    matrix_valid_grid = matrix_valid_grid[indexes_matrix_valid_grid,:]
#
#    
#    ####################################################################
#    
#    '''
#    Now it's necessary :
#        1°)  Interpolate Topos inside the grid grid_dns_x_centralised and grid_dns_y_centralised to the grid of x_piv,y_piv
#            
#        Topos are already selected in the points after pre-processing(centralizing, cutting in x), but the points in space are not the same and are not in the same quantity.
#        Because the spatial sampling frequency is not the same. 
#    '''
#    
#    
#    
#    topos_new_coordinates = np.zeros(shape=(len(x_piv),len(y_piv),*topos_dns.shape[2:]))
#    x = grid_dns_x_centralised
#    y = grid_dns_y_centralised
#    
#    
#    
#    
#    for phi in range(topos_dns.shape[2]):
#        for dim in range(topos_dns.shape[3]):
#            z = topos_dns[:,:,phi,dim].T
#            f = interpolate.interp2d(x, y, z, kind='cubic')
#            znew = f(x_piv, y_piv)
#            topos_new_coordinates[:,:,phi,dim] = znew.T
#    
#        
#    new_distance = x_piv[1]-x_piv[0]
#    return topos_new_coordinates,new_distance


def reduce_and_interpolate_topos_same_as_matlab(topos_dns,grid_dns,MX,dim,slicing,slice_z,u_inf_measured,cil_diameter,center_cil_grid_dns_x_index,\
                                                                         center_cil_grid_dns_y_index,center_cil_grid_PIV_x_distance,\
                                                                         center_cil_grid_PIV_y_distance,distance_of_correlation):
    
    
    '''
    This function finds the grid that is common to DNS and PIV. The function applies a linear interpolation too, 
    because the spatial sampling frequency is different in each case. 
   
    '''
    
    topos_dns = np.reshape(topos_dns,newshape =(MX[0],MX[1],MX[2],topos_dns.shape[-2],dim),order='F') # Reshape topos 
    
    if slicing == True:                             # Apply slice in the data                
        topos_dns = topos_dns[:,:,slice_z,...]
        
    
    '''
    First, we must load one part of the PIV configuration to analise the PIV grid to posterior processing of data PIV and DNS. The data must be measured in
    the same points.
    '''
    file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('wake_Re300_export_190709_0100').joinpath('B0001.dat') # Get the data 
    data = open(str(file))
    datContent = [i.strip().split() for i in data.readlines()]
    data = datContent[4:]
    
    nb_lines = len(data)
    nb_collums = len(data[0])
    
    matrix = np.zeros(shape=(nb_lines,nb_collums))
    
    for i,line in enumerate(data):
        for j,number in enumerate(line):
            matrix[i,j] = number
    
    
    
    ###################### Select the PIV grid ############################## 
        
            
    '''
                --PIV Matrix configuration--
           |x(mm)|y(mm)|vx(m/s)|vy(m/s)|isValid|
    

    '''

    
    valid = matrix[:,4] # Find where the estimation was well done --> isValid==1
    matrix_valid_grid = matrix[np.where((valid==1))[0],:] # Select only the points that were well estimated
    coordinates_x_PIV = np.unique(matrix_valid_grid[:,0]) # Select the grid coordinates in x 
    coordinates_y_PIV = np.unique(matrix_valid_grid[:,1]) # Select the grid coordinates in y
    
    
    '''
    - The center of the cilinder will be in the coordinates --> [center_cil_grid_PIV_x_distance,center_cil_grid_PIV_y_distance]
    - The coordinates must be normalized to be compared with DNS, in this case the new measures are done in (1/D) , D = cilynder diameter in PIV experiments
   
     Example:
         ((xo ; yo) = (-75.60 ; 0.75) )
    
    
    '''
    
    
    coordinates_x_PIV = (coordinates_x_PIV - center_cil_grid_PIV_x_distance)/cil_diameter
    coordinates_y_PIV = (coordinates_y_PIV - center_cil_grid_PIV_y_distance)/cil_diameter
    
    '''
    - We estimated the correlation distance in the observation matrix of PIV, this value is stocked in the variable ''distance_of_correlation'', therefore 
    we need to find the quantity of points in PIV that suffers with border effects from the observer.
    
    number_of_points = distance_of_correlation/spatial_sampling_period_PIV
    
    
    '''
    spatial_sampling_period_PIV = coordinates_x_PIV[1] - coordinates_x_PIV[0]
    number_of_points_PIV = int(distance_of_correlation/spatial_sampling_period_PIV)
    
    coordinates_x_PIV = coordinates_x_PIV[number_of_points_PIV:-(number_of_points_PIV)]
    coordinates_y_PIV = coordinates_y_PIV[number_of_points_PIV:-(number_of_points_PIV)] 
    
    ###################### Select the DNS grid ##############################
    '''
    We must select the DNS grid. After we need to centralize it in the same point; In our case, we had chosen the center of the cilinder as common point. 
    Therefore, we must centralize. The information of the center of the cilynder transformed in index of the grid is stocked in  ''center_cil_grid_dns_x_index''
    and ''center_cil_grid_dns_y_index''.
    '''
    
    coordinates_x_DNS = grid_dns[0] - grid_dns[0][center_cil_grid_dns_x_index]
    coordinates_y_DNS = grid_dns[1] - grid_dns[1][center_cil_grid_dns_y_index]
    
    
    spatial_sampling_period_DNS = coordinates_x_DNS[1] - coordinates_x_DNS[0]
    number_of_points_DNS = int(distance_of_correlation/spatial_sampling_period_DNS)
    
    coordinates_x_DNS = coordinates_x_DNS[number_of_points_DNS:-number_of_points_DNS]
    coordinates_y_DNS = coordinates_y_DNS[number_of_points_DNS:-number_of_points_DNS]
    
    '''
    As we have noticed, the DNS grid is not centered in the cilynder, in other wrds we have more points sampled in one side than another. Hence, 
    it's necessary to cut extra points and assimilate the same amount of points in both sides. 
    
    The coordinates_y_DNS[0] and coordinates_y_DNS[1] are extra, so we'll cut it. 
    '''
    coordinates_y_DNS = coordinates_y_DNS[2:]
    
    
    
    #################### Select DNS grid comparing to PIV grid #####################
    '''
    First: The DNS grid is bigger in X, so we need to decrease the size of it to one point grater than PIV grid. 
    In Y, PIV is bigger, so we need to decrease the grid to one point smaller than DNS grid. 
    
    
    The greater and smaller need to be respected in order to do interpolation from DNS to PIV and not extrapolation from one grid to another. 
    
    
    '''
    
    index_min_x_DNS = np.where((coordinates_x_DNS>coordinates_x_PIV[0]))[0][0]-1
    index_max_x_DNS = np.where((coordinates_x_DNS>coordinates_x_PIV[-1]))[0][0]
    coordinates_x_DNS = coordinates_x_DNS[index_min_x_DNS:index_max_x_DNS+1]
    
       
    index_min_y_PIV = np.where((coordinates_y_PIV>coordinates_y_DNS[0]))[0][0]
    index_max_y_PIV = np.where((coordinates_y_PIV>coordinates_y_DNS[-1]))[0][0]-1
    
    coordinates_y_PIV = coordinates_y_PIV[index_min_y_PIV:index_max_y_PIV+1]
    
    
    ########## Now it's necessary to transform the (Hpiv x topos) that is in PIV sampling frequency to DNS sampling frequency
    
    '''
    In practice, this operation of transform data from one sampling frequency to another is considered one of the Hpiv tasks, 
    the observation matrix that transform data in PIV to DNS. In this case we will transform data sampled in DNS grid to PIV grid,
    because in assimilation step we would need to interpolate data from PIV to DNS every time we assimilate data. 
    '''
    
    original_DNS_x_grid = grid_dns[0] - grid_dns[0][center_cil_grid_dns_x_index]
    original_DNS_y_grid = grid_dns[1] - grid_dns[1][center_cil_grid_dns_y_index]
    
    
    position_x_DNS_begin = np.where((original_DNS_x_grid==coordinates_x_DNS[0]))[0][0]
    position_x_DNS_end   = np.where((original_DNS_x_grid==coordinates_x_DNS[-1]))[0][0]
    
    position_y_DNS_begin = np.where((original_DNS_y_grid==coordinates_y_DNS[0]))[0][0]
    position_y_DNS_end = np.where((original_DNS_y_grid==coordinates_y_DNS[-1]))[0][0]
    
    
    # Getting the information in the smaller DNS grid
    
    topos_dns = topos_dns[position_x_DNS_begin:position_x_DNS_end+1,position_y_DNS_begin:position_y_DNS_end+1,:,:]
    
    
    '''
    Beginning the interpolation
    '''
    nombre_de_modes = topos_dns.shape[2]
    topos_new_coordinates = np.zeros(shape=(len(coordinates_x_PIV),len(coordinates_y_PIV),nombre_de_modes,dim))
    
    for phi in range(topos_dns.shape[2]):
        for dimen in range(dim):
            data = topos_dns[:,:,phi,dimen].T
            f = interpolate.interp2d(coordinates_x_DNS, coordinates_y_DNS, data, kind='linear')
            data_new = f(coordinates_x_PIV, coordinates_y_PIV)
            topos_new_coordinates[:,:,phi,dimen] = data_new.T
    
    
    
  
    
    
    return topos_new_coordinates,coordinates_x_PIV,coordinates_y_PIV
    


def calculate_rotational(topos_Fx,topos_Fy,delta_space,nb_x,nb_y):
    
    '''
    We need to calculate the z curl component of the fluid
    '''
    
    Fx = np.reshape(topos_Fx,(nb_x,nb_y,topos_Fx.shape[1]),order='F')  # Reshape the vector in x direction
    Fy = np.reshape(topos_Fy,(nb_x,nb_y,topos_Fy.shape[1]),order='F')  # Reshape the vector in y direction
    
    Field_dFy_dx = np.zeros((Fy.shape[0]-2,Fy.shape[1],Fy.shape[2]))   # Create vector to stock dFy/dx
    
    Field_dFx_dy = np.zeros((Fx.shape[0],Fx.shape[1]-2,Fx.shape[2]))   # Create vector to stock dFx/dy 
    
    
    for component in range(Fy.shape[-1]):                                                          # Get Vector in y direction for all components(number of modes if Topos)
#        print(component)
        for line in range(nb_y):                                                                   # Go through all the lines calculating the derivative 
            i=0
            for point in range(1,nb_x-1):                                                          # For each line calculate the derivative in all y
                dFy_dx = (Fy[point+1,line,component]-Fy[point-1,line,component])/(2*delta_space)   # Derivative
                Field_dFy_dx[i,line,component] = dFy_dx                                            # Stocks 
                i = i+1
                
    for component in range(Fx.shape[-1]):                                                             # Get Vector in y direction for all components(number of modes if Topos)
#        print(component)
        for collum in range(nb_x):                                                                    # Go through all the lines taking the derivative 
            i=0
            for point in range(1,nb_y-1):                                                             # For each collumn calculate the derivative in all x
                dFx_dy = (Fx[collum,point+1,component]-Fx[collum,point-1,component])/(2*delta_space)  # Derivative
                Field_dFx_dy[collum,i,component] = dFx_dy                                             # Stocks
                i = i+1
    
    curl_Z = Field_dFy_dx[:,1:-1,:] - Field_dFx_dy[1:-1,...]   # Take the rotational
    
    return curl_Z
    
    
    
#%%                                          Begin the main_from_existing_ROM that constrols all the simulation
    
def main_from_existing_ROM(nb_modes,threshold,type_data,nb_period_test,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt,n_particles):#nb_modes,threshold,type_data,nb_period_test,no_subampl_in_forecast,reconstruction,adv_corrected,modal_dt):
    
    
    ######################################----PARAMETERS TO CHOOSE----############################################
    # Parameters choice
    param_ref = {}
    param_ref['n_simu'] = 100               # Number of simulations steps in time
    param_ref['N_particules'] = n_particles # Number of particles to select  
    beta_1 = 0.1                            # beta_1 is the parameter that controls the noise to create virtual observation beta_1 * np.diag(np.sqrt(lambda))
    beta_2 = 0.1                            # beta_2 is the parameter that controls the  noise in the initialization of the filter
    beta_3 = 1                              # beta_3 is the parameter that controls the impact in the model noise -> beta_3 * pchol_cov_noises 
    beta_4 = 5                              # beta_4 is the parameter that controls the time when we will use the filter to correct the particles
    N_threshold = 20                        # Effective sample size in the particle filter
    nb_mutation_steps = 20                  # Number of mutation steps in particle filter 
    pho = 0.9998                            # Constant that constrol the balance in the new brownian and the old brownian in particle filter
#    L = 0.75*0.00254/(32*10**(-3))         # Incertitude associated with PIV data estimated before. It was used in the Sigma matrix estimation. 
    std_space = 0.0065/(32*10**-3)          # Correlation distance in PIV measures
    assimilate = 'real_data'                # The data that will be assimilated
    mask_obs = True                         # Activate spatial mask in the observed data
    subsampling_PIV_grid_factor = 3         # Subsampling constant that will be applied in the observed data, i.e if 3 we will take 1 point in 3 
    only_load = False                       # If False Hpiv*Topos will be calculated, if True and calculated before it will be loaded 
    slicing = True                          # If True we will select one slice to assimilate data, because with 2d2c PIV we have only one slice.
    slice_z = 30                            # The slice that will be assimilated: It should be 30 because the Hpiv*topos calculated in matlab take in account the slice 30
    data_assimilate_dim = 2                 # In this experiments we assimilate 2D data, and in the case Reynolds=300, the vector flow in the z direction will be ignored.
    u_inf_measured = 0.388                  # The PIV measured velocity (See the .png image in the respective PIV folder with all measured constants). It must be represented in m/s
    cil_diameter = 12                       # Cylinder diameter in PIV experiments. It must be in mm. (See the .png image in the respective PIV folder with all measured constants).
    dt_PIV = 0.080833                       # Temporal step between 2 consecutive PIV images. (See the .png image in the respective PIV folder with all measured constants).    
    center_cil_grid_dns_x_index = 60        # Index that represents the center of the cylinder in X in the DNS grid
    center_cil_grid_dns_y_index = 49        # Index that represents the center of the cylinder in Y in the DNS grid
    Re=300                                  # Reynolds constant
    center_cil_grid_PIV_x_distance = -75.60 # Center of the cylinder in X in the PIV grid (See the .png image in the respective PIV folder with all measured constants). It ust be in mm.
    center_cil_grid_PIV_y_distance = 0.75   # Center of the cylinder in Y in the PIV grid (See the .png image in the respective PIV folder with all measured constants). It ust be in mm.
    SECONDS_OF_SIMU = 1                   # We have 331 seconds of real PIV data for reynolds=300 beacuse we have 4103 files. --> ( 4103*0.080833 = 331)

    number_of_PIV_files = int(SECONDS_OF_SIMU/dt_PIV) + 1                                                    # Number of PIV files to load
    vector_of_assimilation_time = np.arange(start=dt_PIV,stop=(number_of_PIV_files+1)*dt_PIV,step=dt_PIV) # Construct the moments that can be assimilated. 
    sub_sampling_PIV_data_temporaly = True                                                                   # We can choose not assimilate all possible moments(time constraints or filter performance constraints or benchmark constraints or decorraltion hypotheses). Hence, select True if subsampling necessary 
    factor_of_PIV_time_subsampling = 3                                                                       # The factor that we will take to subsampled PIV data. 
    vector_of_assimilation_time = vector_of_assimilation_time[::factor_of_PIV_time_subsampling]              # Using the factor to select the moments that we will take to assimilate
    plt_real_time = True                                                                                     # It can be chosen to plot chronos evolution in real time or only at the end of the simulation
    x0_index = 6                                                                                             # Parameter necessary to chose the grid that we will observe(i.e if 6 we will start the select the start of the observed grid in the 6th x index, hence we will reduce the observed grid).
    nbPoints_x = 10                                                                                          # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
    y0_index = 30                                                                                            # Parameter necessary to chose the grid that we will observe(i.e if 30 we will start the observed grid in the 30th y index, hence we will reduce the observed grid).
    nbPoints_y = 15                                                                                          # Number of points that we will take in account in the observed grid. Therefore, with this two parameters we can select any possible subgrid inside the original PIV/DNS grid to observe.
    #################################### ----------------------------------------------------------------------- ###################################
    
    
    
    #%%  Parameters already chosen
    
    #   !!!! Do not modify the following lines  !!!!!!
#    coef_correctif_estim = {}
#    coef_correctif_estim['learn_coef_a'] = True # true or false
#    coef_correctif_estim['type_estim'] = 'vector_b' #  'scalar' 'vector_z' 'vector_b' 'matrix'
#    coef_correctif_estim['beta_min'] =  math.inf # -inf 0 1
    
    
    current_pwd = Path(__file__).parents[1] # Select the path
    folder_results = current_pwd.parents[0].joinpath('resultats').joinpath('current_results') # Select the current results path
    folder_data = current_pwd.parents[0].joinpath('data') # Select the data path
    
    
    param_ref['folder_results'] = str(folder_results) # Stock folder results path
    param_ref['folder_data'] = str(folder_data)       # Stock folder data path
    
    
    modal_dt_ref = modal_dt # Define modal_dt_ref
     
    
    
    #%% Get data
    
    # On which function the Shanon ctriterion is used
    test_fct = 'b'  # 'b' is better than db
    a_t = '_a_cst_' 
    
    ############################ Construct the path to select the model constants I,L,C,pchol and etc.
    
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
    I_sto,L_sto,C_sto,I_deter,L_deter,C_deter,plot_bts,pchol_cov_noises,bt_tot,param = convert_mat_to_python(str(file_res)) # Call the function and load the matlab data calculated before in matlab scripts.
    param['decor_by_subsampl']['no_subampl_in_forecast'] = no_subampl_in_forecast                                           # Define the constant
    
    
    
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
    bt_tot = bt_tot[:param['N_test'] + 1,:]                # Ref. Chronos in the DNS case
    bt_tronc=bt_tot[0,:][np.newaxis]                       # Define the initial condition as the reference
    
    param['dt'] = param['dt']/n_simu                       # The simulation time step is dependent of the number of time evolution steps between the param['dt'],therefore the new param['dt'] is divided by the number of evolution steps 
    param['N_test'] = param['N_test'] * n_simu             # Number of model integration steps is now the number of steps before times the number of integration steps between two old steps
    
    
#    Reconstruction in the deterministic case
#    bt_forecast_deter = bt_tronc
#    for index in range(param['N_test']):
#        bt_forecast_deter = np.vstack((bt_forecast_deter, evol_forward_bt_RK4(ILC_a_cst['deter']['I'],ILC_a_cst['deter']['L'],ILC_a_cst['deter']['C'],param['dt'],bt_forecast_deter)))
    
#    Reconstruction in the stochastic case
#    bt_forecast_sto = bt_tronc
#    for index in range(param['N_test']):
#        bt_forecast_sto = np.vstack((bt_forecast_sto,evol_forward_bt_RK4(ILC_a_cst['modal_dt']['I'],ILC_a_cst['modal_dt']['L'],ILC_a_cst['modal_dt']['C'],param['dt'],bt_forecast_sto)))
    
    
    
    
#    Reconstruction in the stochastic case
    lambda_values = param['lambda'][:,0] # Define the constant lambda (The integral in one period of the square temporal modes )
    
#    lambda_values = lambda_values[0]*np.ones(lambda_values.shape)
    
    
    bt_MCMC = np.tile(bt_tronc.T,(1,1,param['N_particules']))   # initializes the chronos of all particules equally
    shape = (1,bt_tronc.shape[1],int(param['N_particules']))    # Define the shape of this vector containing all the chronos of all particles
    
    
    
#    bt_MCMC[:,:,:] =  bt_MCMC + beta_2*np.tile(lambda_values[...,np.newaxis],(1,1,bt_MCMC[:,:,:].shape[2]))*np.random.normal(0,1,size=shape)
    bt_MCMC[:,:,:] = beta_2*np.tile(lambda_values[...,np.newaxis],(1,1,bt_MCMC[:,:,:].shape[2]))*np.random.normal(0,1,size=shape) # Initialise the chronos paricles randomly and dependent of the lambda values 
    
    bt_fv = bt_MCMC.copy()                                              # Define bt_fv 
    bt_m = np.zeros((1,int(param['nb_modes']),param['N_particules']))   # Define bt_m
    iii_realization = np.zeros((param['N_particules'],1))               # Define iii_realization that is necessary if any explosion in the simulation
    
    
        
   #%%  Loading matrices H_PivTopos calculated before, select the data in the new PIV grid and load sigma_inverse in the PIV space
   
   
#      LOAD TOPOS
    print('Loading Topos')
    path_topos = Path(folder_data).parents[1].joinpath('data').joinpath('mode_'+type_data+'_'+str(nb_modes)+'_modes') # Topos path 
    topos_data = hdf5storage.loadmat(str(path_topos))                                                                 # Load topos
    topos = topos_data['phi_m_U']                                                                                     # Select Topos          
    MX = param['MX'][0,:]                                                                                             # Select DNS grid
    
    if topos.shape[-1]==3:                        # Select only the u and v vector field.
        topos = topos[...,:data_assimilate_dim]   # Selecting...
    
    dim = topos.shape[-1]                                                                              # Define the vectorial field dimension
    topos_l = np.transpose(topos,(0,2,1))                                                              # Rearrange dimensions 
    topos_l = np.reshape(topos_l,(int(topos_l.shape[0]*topos_l.shape[1]),topos_l.shape[2]),order='F')  # Reshape the topos, the last dimensions being the number of resolved modes plus one and the first dimensions is (Nx * Ny * dim)
    grid = param['MX'][0]                                                                              # Define the DNS grid
    distance = param['dX'][0,0]                                                                        # Define the spatial space between 2 samples in DNS grid 
    matrix_H,number = calculate_H_PIV(topos_l,distance,grid,std_space,only_load,dim,slicing,slice_z)   # Apply spatial filter in the topos. The filtr was estimated before and is based in the PIV measures
    
    print('Calculating PIV mask and applying on the Topos')
    matrix_H = np.transpose(matrix_H,(0,1,2,4,3))                                                                                                                                # Rearrange  matrix_H
    matrix_H = np.reshape(matrix_H,(int(matrix_H.shape[0]*matrix_H.shape[1]*matrix_H.shape[2]),matrix_H.shape[3],matrix_H.shape[4]),order='F')                                   # Reshape matrix_H  
    if assimilate == 'real_data':                                                                                                                                                # Select if real data to transform Hpiv_Topos from space DNS to PIV
        topos_new_coordinates,coordinates_x_PIV,coordinates_y_PIV = reduce_and_interpolate_topos_same_as_matlab(matrix_H,param['grid'][0],MX,data_assimilate_dim,slicing,\
                                                                                                             slice_z,u_inf_measured,cil_diameter,center_cil_grid_dns_x_index,\
                                                                                                             center_cil_grid_dns_y_index,center_cil_grid_PIV_x_distance,\
                                                                                                             center_cil_grid_PIV_y_distance,std_space)
    
              
    '''
    The Sigma_inverse matrix was calculated before in the space H_piv, so we need just to load it.
    
        - The folder ''data_PIV'' contains all files related to measured data 
        - Therefore we need to search HSigSigH_PIV..... and load it here. It was calculated before in matlab.
        
                                         ---------------------------VERY IMPORTANT--------------------------------
                                         
            - THIS MATRIX IS DEPENDENT OF THE PIV MEASURED NOISE. THEREFORE THE MATRIX L. BEING SIGMA = [HSigSigH + LL]; 
            - THE VALUE USED HERE WAS ESTIMATED IN THE IRSTEA DATA AND REPRESENTS 6% OF THE AMPLITUDE.
            - THE NOISE IS UNCORRELATED IN TIME AND IN SPACE. SUBSAMPLING SPATIALLY AND TEMPORALLY CAN INCREASE THE POSSIBILITY OF BE TRUE.
    
    '''
    
    path_Sigma_inverse = Path(__file__).parents[3].joinpath('data_PIV').joinpath('HSigSigH_PIV_'+type_data+'_'+str(param['nb_modes'])+'_modes_a_cst_threshold_0_'+str(threshold)[2:])  # Load Sigma_inverse
    Sigma_inverse_data = hdf5storage.loadmat(str(path_Sigma_inverse)) # Select Sigma_inverse
    
    
    '''
    - The subgrid that Valentin used in the Matlab code was not exactly the original PIV grid. Therefore we have not found the same grid. Therefore, the indexes 3->-4 and 1->-1 cutting 'coordinates_x_PIV' and 
    'coordinates_y_PIV' transforms my estimated grid in the same grid estimated by valentin.
    '''
    coordinates_x_PIV = coordinates_x_PIV[3:-4]                            # Transforming the grid in x 
    coordinates_y_PIV = coordinates_y_PIV[1:-1]                            # Transforming the grid in y
    topos_new_coordinates = topos_new_coordinates[3:-4,1:-1,:,:]           # Selecting the data in valentin grid
    topos_new_coordinates = np.transpose(topos_new_coordinates,(0,1,3,2))  # Transposing the topos 
    

    
    Hpiv_Topos = np.reshape(topos_new_coordinates,(int(topos_new_coordinates.shape[0]*topos_new_coordinates.shape[1]*topos_new_coordinates.shape[2]),topos_new_coordinates.shape[3]),order='F') # The topos that we have estimated reshaped to posterior matrix multiplications

#    np.save('Hpiv_Topos.npy',Hpiv_Topos) # If save, it will save the topos in a numpy file in the same folder of the script.
    
    #%%  Define and apply mask in the observation
    '''
    In the lines below we define and apply the observation mask. It allows define where we'll observe in the PIV grid and
    if necessary with smaller spatial grid. 
    
    Matrices that should receive M_mask:
        - Y_obs
        - H_piv_Topos
        - Sigma_inverse
    
    The information about the grid points in x and y spatially represented are stocked in 
    coordinates_x_PIV and coordinates_y_PIV, respectivelly.
    
    ''' 
    if mask_obs == True:   # If we must select a smaller grid inside the observed grid. 
        
        xEnd_index = x0_index + nbPoints_x*subsampling_PIV_grid_factor  # Define the last x index of the new grid
        yEnd_index = y0_index + nbPoints_y*subsampling_PIV_grid_factor  # Define the last y new grid index
        
        if (yEnd_index-subsampling_PIV_grid_factor)>(len(coordinates_y_PIV-1)) or  (xEnd_index-subsampling_PIV_grid_factor)>(len(coordinates_x_PIV-1)): # Checks if the points are inside the observed grid
            print('Error: grid selected is bigger than the observed')
            sys.exit()
        
        coordinates_x_PIV_with_MASK = coordinates_x_PIV[x0_index:xEnd_index:subsampling_PIV_grid_factor]  # Selecting the new grid 
        coordinates_y_PIV_with_MASK = coordinates_y_PIV[y0_index:yEnd_index:subsampling_PIV_grid_factor]  # Selecting the new grid
        
        '''
        We need to construct the mask to select the coefficients in the matrices.
        '''
        Mask_x = np.zeros(len(coordinates_x_PIV),dtype = np.int8)    # Construct the mask x
        Mask_x[x0_index:xEnd_index:subsampling_PIV_grid_factor] = 1  # Define the points of the new grid as 1
        
        Mask_y = np.zeros(len(coordinates_y_PIV),dtype = np.int8)    # Construct the mask y
        Mask_y[y0_index:yEnd_index:subsampling_PIV_grid_factor] = 1  # Define the points of the new grid as 1
        
        ###############################--Begin the mask construction--#############
        if Mask_y[0]==0:                                       # If the first collum dont  belongs to the grid
            Mask_final = np.zeros(len(Mask_x),dtype = np.int8) # Add a zeros to the mask
        else:                                                  # If it belongs
            Mask_final = Mask_x.copy()                         # Add the Mask x 
        
        for line in Mask_y[1:]:                                                                     # The process continues to y>1 until the last column
            if line==0:
                Mask_final = np.concatenate((Mask_final,np.zeros(len(Mask_x),dtype = np.int8)))
            else: 
                Mask_final = np.concatenate((Mask_final,Mask_x.copy()))
                
                
        Mask_final = np.concatenate((Mask_final,Mask_final))                                        # It must be concatenated with himself because we are working with 2 dimensions
        Mask_final_bool = Mask_final.astype(bool)                                                   # Transform the data inside in boolean. 1->True and 0->False
    
        print('The coordinates that will be observed: ')
        print('The x coordinates: '+str(coordinates_x_PIV_with_MASK))
        print('The y coordinates: '+str(coordinates_y_PIV_with_MASK))
#    
    #%%   Calculate Sigma_inverse
    
    print('Loading Sigma and multipling likelihood matrices')
    Sigma_inverse = Sigma_inverse_data['inv_HSigSigH'][:,0,:,:]                             # Load Sigma inverse 
    Sigma_inverse = Sigma_inverse[Mask_final_bool[:Sigma_inverse.shape[0]],:,:].copy()      # SelectSigma inverse in the mask that we observe    
    
    ##### Transform this matrix in a square matrix
    nb_points = Sigma_inverse.shape[0]                                                      # Number of points in the grid
    nb_dim = Sigma_inverse.shape[1]                                                         # Dimension
    
    
    
    '''
    The sigma inversed must be transformed in a matrix 
        - It contains the inversed matrix of correlations that is uncorrelated in space and in time. The correlation is between dimensions. 
        
    '''
    Sigma_inverse_squared = np.zeros((int(nb_points*nb_dim),int(nb_points*nb_dim)))     # Create a matrix to stock the data     
    
    for line in range(int(nb_points)):                                                  # To all spatial samples we create the first part of the matrix that contains the correlation of Vx
        Sigma_inverse_squared[line,line] = Sigma_inverse[line,0,0]                      # Correlation of Vx with Vx
        Sigma_inverse_squared[line,line+nb_points] =  Sigma_inverse[line,0,1]           # Correlation of Vx with Vy 
       
    for line in range(int(nb_points)):                                                  # Now the second part of the matrix with Vy
        Sigma_inverse_squared[line+nb_points,line] = Sigma_inverse[line,1,0]            # Correlation of Vy with Vx
        Sigma_inverse_squared[line+nb_points,line+nb_points] =  Sigma_inverse[line,1,1] # Correlation of Vy with Vy
    
    
  
    ######## Applying final mask after define observation window
    Hpiv_Topos = Hpiv_Topos[Mask_final_bool,:].copy()                   

    
    
    # Calculating necessary matrices
    K = Sigma_inverse_squared @ Hpiv_Topos   # We define K as the sigma inversed matrix times the Hpiv_Topos matrix
    Hpiv_Topos_K = Hpiv_Topos.T @ K          # The Hpiv_Topos times K is necessary too
    
    
#%%
    
    
    # LOAD dns simulation 
#    if type_data == 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated':
#        folder_DNS = 'folder_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks'
#        for number_of_file in range(81,81+number_of_files):
##        number_of_file = 81 # at least 81 because the others are used to train and max is 100
#        
#            file = 'file_DNS300_inc3d_3D_2017_04_02_'+ str(number_of_file)
#            Path_file_fluid_flow_DNS = Path(folder_data).parents[1].joinpath('data').joinpath(folder_DNS).joinpath(file)
#            data_DNS = hdf5storage.loadmat(str(Path_file_fluid_flow_DNS))
#            
#            if number_of_file == 81:
#                dt = data_DNS['dt'][0,0]
#                champ = data_DNS['U']
#                del data_DNS
#            else:
#                champ = np.concatenate((champ,data_DNS['U']),axis=3)
#                del data_DNS
#            
#            print('File number '+str(number_of_file)+' loaded')
        
#    elif type_data == 'DNS100_inc3d_2D_2018_11_16_blocks_truncated':
#        folder_DNS = 'folder_DNS100_inc3d_2D_2018_11_16_blocks'
#        number_of_file = 18 # at least 16 because the others are used to train and max is 20
#        file = 'file_DNS100_inc3d_2018_11_16_run_'+ str(number_of_file)
    
#    Path_file_fluid_flow_DNS = Path(folder_data).parents[1].joinpath('data').joinpath(folder_DNS).joinpath(file)
#    data_DNS = hdf5storage.loadmat(str(Path_file_fluid_flow_DNS))
#    champ = data_DNS['U']
#    dt = data_DNS['dt'][0,0]
#    param['dt'] = dt/n_simu # dt is coming from DNS before subsampling of topos 
#    param['N_test'] = n_simu*(champ.shape[3]-1)
#    
    ######################################################################
    
    
#    particles_estimate = np.zeros((1,bt_MCMC.shape[1]))
#    n_samples = int(5*1/param['dt'])
#    period_in_samples = int(5*1/param['dt'])
#    period = False
#    weigths_time_past = np.ones((bt_MCMC.shape[2]))/bt_MCMC.shape[2]
#    n,nb_pc1 = bt_MCMC.shape[1:]
#    shape_noise = (1,(n+1)*n,nb_pc1)
#    noises = np.zeros(shape=shape_noise)
   
#    observations_to_save = bt_tronc + beta_2*lambda_values[np.newaxis,...]*np.random.normal(0,1,size=(1,bt_tronc.shape[1]))
    
#    if assimilate_DNS == False:
#        factor = 3
#    else: 
#        factor = 1
    ###################################--calculate important matrices to assimilation #####################
    
#    reconstruc_flow = calculate_residue_and_return_reconstructed_flow(topos_l,champ,bt_tot)
#    tensor_var = load_cov_tensor(Path(folder_data).parents[1].joinpath('data').joinpath(Path(param['name_file_diffusion_mode']).name))
    
    
    
    
    
#    variance_uxx = tensor_var[:,0,0]
#    variance_uxy = tensor_var[:,0,1]
#    variance_uxz = tensor_var[:,0,2]
#    
#    variance_uyx = tensor_var[:,1,0]
#    variance_uyy = tensor_var[:,1,1]
#    variance_uyz = tensor_var[:,1,2]
#    
#    variance_uzx = tensor_var[:,2,0]
#    variance_uzy = tensor_var[:,2,1]
#    variance_uzx = tensor_var[:,2,2]
    
    
    
#    sigma = np.concatenate((variance_ux,variance_uy,variance_uz))[...,np.newaxis]
    
#    if mask == True:
#        for i in range(len(param['MX'][0,:])):
#            param['MX'][0,i] = math.ceil(param['MX'][0,i]/subsampling_grid)
#            
#            
#            
##        matrix_H = matrix_H.reshape((champ.shape[0],champ.shape[1],champ.shape[2],champ.shape[4],matrix_H.shape[1]),order='F')
##        sigma = sigma.reshape((champ.shape[0],champ.shape[1],champ.shape[2],champ.shape[4],sigma.shape[1]),order='F')
#        topos_l = topos_l.reshape((champ.shape[0],champ.shape[1],champ.shape[2],champ.shape[4],topos_l.shape[1]),order='F')
#        
#        
##        matrix_H = matrix_H[::subsampling_grid,::subsampling_grid,::subsampling_grid,:,:]
##        sigma = sigma[::subsampling_grid,::subsampling_grid,::subsampling_grid,:,:]
#        champ = champ[::subsampling_grid,::subsampling_grid,::subsampling_grid,:]
#        topos_l = topos_l[::subsampling_grid,::subsampling_grid,::subsampling_grid,:]
#        
##        matrix_H = matrix_H.reshape((matrix_H.shape[0]*matrix_H.shape[1]*matrix_H.shape[2]*matrix_H.shape[3],matrix_H.shape[4]),order='F')
##        sigma = sigma.reshape((sigma.shape[0]*sigma.shape[1]*sigma.shape[2]*sigma.shape[3],sigma.shape[4]),order='F')
#        topos_l = topos_l.reshape((topos_l.shape[0]*topos_l.shape[1]*topos_l.shape[2]*topos_l.shape[3],topos_l.shape[4]),order='F')
  
#    np.save('grid.npy',param['grid'][0])
#    if dim==2:
#        grid = np.concatenate((topos.shape[:2],np.array([1])))
#    elif dim==3:
#        grid = topos.shape[:3]
        
    
    
    
    
#    teste_matrix = np.ravel(matrix_H,order='F')
#    error = np.subtract(teste_topos, teste_matrix)
#    pos = np.where((error!=0))[0]
    
#    topos_l_res = topos_l.reshape((grid[0],grid[1],grid[2],dim,nb_modes+1),order='F')

#    topos_l_res = topos_l_res[number:-number,number:-number,...]
#    topos_l_res_ens = topos_l_res.reshape((int(topos_l_res.shape[0]*topos_l_res.shape[1]*topos_l_res.shape[2]*topos_l_res.shape[3]),int(topos_l_res.shape[4])),order='F')
#    topos_l_res_ux = topos_l_res[number:-number,:,slice_z:slice_z+1,0,:]
#    topos_l_res_uy = topos_l_res[:,number:-number,slice_z:slice_z+1,1,:]
#    topos_l_res_uz = topos_l_res[:,:,slice_z:slice_z+1,2,:]
#    
#    topos_l_res_ux = topos_l_res_ux.reshape((topos_l_res_ux.shape[0]*topos_l_res_ux.shape[1]*topos_l_res_ux.shape[2],topos_l_res_ux.shape[3]),order='F')
#    topos_l_res_uy = topos_l_res_uy.reshape((topos_l_res_uy.shape[0]*topos_l_res_uy.shape[1]*topos_l_res_uy.shape[2],topos_l_res_uy.shape[3]),order='F')
#    topos_l_res_uz = topos_l_res_uz.reshape((topos_l_res_uz.shape[0]*topos_l_res_uz.shape[1]*topos_l_res_uz.shape[2],topos_l_res_uz.shape[3]),order='F')
#    
#    topos_l_res_ens = np.concatenate((topos_l_res_ux,topos_l_res_uy,topos_l_res_uz))
#    
#    error_matrix = np.subtract(matrix_H,topos_l_res_ens)
#    pos = np.where((error_matrix!=0))[0]
#    error = np.sum(np.power(error_matrix,2),axis=0)
    
    
    
    
#    dt_optimal = param['dt']*param_ref['n_simu']
#    Sigma_inversed_diagonal = calculate_noise_covariance_tensor(tensor_var,L,param['dX'],param['MX'],std_space,subsampling_grid,dim,dt_optimal,slicing,slice_z)

    
    
#    if Sigma_inversed_diagonal.shape[1] == 1:
#        K = (np.multiply(Sigma_inversed_diagonal,matrix_H)).T
#    else:
#        K = (np.matmul(Sigma_inversed_diagonal,matrix_H)).T
#
#    
##    u, s, vh = np.linalg.svd(K)
#    
#    
#    Hr_K = np.matmul(matrix_H.T,K.T)
    
#   ##########################  
    
#%%                    LOAD   Data PIV 

    file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('wake_Re'+str(Re)).joinpath('B'+str(1).zfill(4)+'.dat')   # The path to load PIV data
    data = open(str(file))                                                                                                     # Open the PIV data  
    datContent = [i.strip().split() for i in data.readlines()]                                                                 # Reading the data 
    data = datContent[4:]                                                                                                      # Getting the data PIV
    
    nb_lines = len(data)                           # lines amount
    nb_collums = len(data[0][2:4])                 # Commumns amount 
    matrix = np.zeros(shape=(nb_lines,nb_collums)) # Creating matrix to stock data 
    
    '''
    We will select the first data in 0.080833 and after we will load the files taking in account the factor of subsampling and the amount of files. 
    '''
    for i,line in enumerate(data):  # Select the data in the first PIV file 
        for j,number in enumerate(line[2:4]):
            matrix[i,j] = number
    
    matrix_data_PIV_all_data = matrix[Sigma_inverse_data['mask'][:,0],:].copy()[np.newaxis,...] # Select the PIV data in the first mask calculated (The mask inside PIV and DNS)    
    print('Loading PIV data')
    print(number_of_PIV_files)

    for nb_file in range(1,(number_of_PIV_files+1),factor_of_PIV_time_subsampling)[1:]:                                                    # Loading the other files as defined in the start of this function
        print(nb_file)
        file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('wake_Re'+str(Re)).joinpath('B'+str(nb_file).zfill(4)+'.dat') # Path to file
        data = open(str(file))                                                                                                         # Open the file       
        datContent = [i.strip().split() for i in data.readlines()]                                                                     # Read the data                                                                                                                                                                              
        data = datContent[4:]                                                                                                          # Getting the data PIV 
        
        
        matrix = np.zeros(shape=(nb_lines,nb_collums)) # Define the matrix to stock the data  
        for i,line in enumerate(data):                 # Decode the information and save it as a matrix
            for j,number in enumerate(line[2:4]):
                matrix[i,j] = number
        
    
        matrix_data_PIV_all_data = np.concatenate((matrix_data_PIV_all_data,matrix[Sigma_inverse_data['mask'][:,0],:].copy()[np.newaxis,...]),axis=0)  # Save the matrix inside the matrix of all the PIV data
    
    
    # Normalizing  measured data
    matrix_data_PIV_all_data = matrix_data_PIV_all_data/u_inf_measured  # Normalizing the PIV data to compare with DNS 
#    np.save('Data_piv.npy',matrix_data_PIV_all_data)                   # If necessary to save(This will be saved as numpy array in this folder.)
    
    '''
    We need to apply the same observation mask on the observed data. Because the new mask is defined to control where we will observe inside the PIV window
    '''
    matrix_data_PIV_all_data = matrix_data_PIV_all_data[:,Mask_final_bool[:matrix_data_PIV_all_data.shape[1]],:].copy()  
    
    #%% Begin propagation and assimilation
    pchol_cov_noises = beta_3*pchol_cov_noises                           # Cholesky de la matrix de covariance                          
    original_dt_simu = param['dt']                                       # Time simulation step  
    assimilate_PIV = False                                               # Flag to control assimilation moments
    nb_assim = 0                                                         # Flag to count the assimilation steps 
    next_time_of_assimilation = vector_of_assimilation_time[nb_assim]    # Control de next moment that a obs will be available
    index_of_filtering = []                                              # Control de index of assimilation
    time = [0]                                                           # The time of assimilation
    index_pf = [0]                                                       # Flag to control de noise in the past until now to mutation steps in Metropolis-Hastings
    
    # Defining figure to plot if real data is True 
    if plt_real_time==True:
        plt.ion()
        fig = plt.figure(0)
        plt.rcParams['axes.grid'] = True
        
        
        ax_1 = fig.add_subplot(2,2,1)
        ax_1.set_ylim([-10, 10])
        
        ax_2 = fig.add_subplot(2,2,2)
        ax_2.set_ylim([-10, 10])
        
        ax_3 = fig.add_subplot(2,4,5)
        ax_3.set_ylim([-10, 10])
        
        ax_4 = fig.add_subplot(2,4,6)
        ax_4.set_ylim([-10, 10])
        
        ax_5 = fig.add_subplot(2,4,7)
        ax_5.set_ylim([-10, 10])
        
        ax_6 = fig.add_subplot(2,4,8)
        ax_6.set_ylim([-10, 10])
        
        
        
        quantiles_now = np.quantile(bt_MCMC[-1,:,:],q=[0.025,0.975],axis=1)
        particles_mean_now = np.mean(bt_MCMC[-1,:,:],axis=1)
        
        line11, = ax_1.plot(time[-1], particles_mean_now[0], 'b-',label = 'Particles mean')
        line12  = ax_1.fill_between([0], quantiles_now[0:1,0],quantiles_now[1:2,0], color='gray')
        line13,  = ax_1.plot([0],[-2*1],'r.',label = 'Assimilate True')
       
        line21, = ax_2.plot(time[-1], particles_mean_now[1], 'b-',label = 'Particles mean')
        line22 = ax_2.fill_between([0], quantiles_now[0:1,1],quantiles_now[1:2,1], color='gray')
        line23,  = ax_2.plot([0],[-2*1],'r.',label = 'Assimilate True')
      
        line31, = ax_3.plot(time[-1], particles_mean_now[2], 'b-',label = 'Particles mean')
        line32 = ax_3.fill_between([0], quantiles_now[0:1,2],quantiles_now[1:2,2], color='gray')
        line33,  = ax_3.plot([0],[-2*1],'r.',label = 'Assimilate True')
        
        line41, = ax_4.plot(time[-1], particles_mean_now[3], 'b-',label = 'Particles mean')
        line42 = ax_4.fill_between([0], quantiles_now[0:1,3],quantiles_now[1:2,3], color='gray')
        line43,  = ax_4.plot([0],[-2*1],'r.',label = 'Assimilate True')
    
        line51, = ax_5.plot(time[-1], particles_mean_now[4], 'b-',label = 'Particles mean')
        line52 = ax_5.fill_between([0], quantiles_now[0:1,4],quantiles_now[1:2,4], color='gray')
        line53,  = ax_5.plot([0],[-2*1],'r.',label = 'Assimilate True')
       
        line61, = ax_6.plot(time[-1], particles_mean_now[5], 'b-',label = 'Particles mean')
        line62 = ax_6.fill_between([0], quantiles_now[0:1,5],quantiles_now[1:2,5], color='gray')
        line63,  = ax_6.plot([0],[-2*1],'r.',label = 'Assimilate True')
        
        
        
        ax_1.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(1)+'$'+' amplitude')
        ax_1.legend()
        ax_2.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(2)+'$'+' amplitude')
        ax_2.legend()
        ax_3.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(3)+'$'+' amplitude')
        ax_3.legend()
        ax_4.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(4)+'$'+' amplitude')
        ax_4.legend()
        ax_5.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(5)+'$'+' amplitude')
        ax_5.legend()
        ax_6.set(xlabel="Time(sec)", ylabel='Chronos '+r'$b_'+str(6)+'$'+' amplitude')
        ax_6.legend()
    

    
    
    
                   
    ################################ Start temporal integration ###################################
    for index in range(param['N_test']): # Set the number of integration steps

        ##### Model integration of all particles
        val0,val1,val2,noises_centered = evol_forward_bt_MCMC(ILC_a_cst['modal_dt']['I'],\
                                                        ILC_a_cst['modal_dt']['L'],\
                                                        ILC_a_cst['modal_dt']['C'],\
                                                        pchol_cov_noises,param['dt'],\
                                                        bt_MCMC[-1,:,:],bt_fv[-1,:,:],\
                                                        bt_m[-1,:,:],mutation=False,noise_past=0,pho=0)
        time.append(param['dt']+time[-1])
        #########################################----------------------#############################################
        #########################################--PARTICLE FILTERING--#############################################

#        if (index+1)%(int(period_in_samples*beta_4))== 0:
#            period = True
#            print('Index of activating filtering: '+str(index))
        
#        if ((index+1))%(int(factor*n_simu*beta_4)) == 0:
#            period = True
#            print('-----------------------------STARTING PARTICLE FILTERING---------------------------')
#            print('Index of activating filtering: '+str(index))
#        period = True
        
    
        
        
        
        if (assimilate_PIV==True):                      # The Flag assimilate_PIV control the moments that we can assimilate data
            index_of_filtering.append(index)            # Stock the assimilation index
            print('Index of filtering: '+str(index))    
            index_pf.append(index+1)                    # Stock the assimilation index to control noise and past particles

#            obs = bt_tot[ int((index+1)/n_simu),:][...,np.newaxis]
#            index_assimilation = (index+1)/(n_simu)
#            if assimilate_DNS == False:
#                obs = reconstruc_flow[:,:,:,:,int(index_assimilation)]
#                
#                
#            else:
#                obs = champ[:,:,:,int(index_assimilation),:]
            
            # Define the obs and reshape it
            obs = np.reshape(matrix_data_PIV_all_data[nb_assim,:,:],(matrix_data_PIV_all_data[nb_assim,:,:].shape[0]*matrix_data_PIV_all_data[nb_assim,:,:].shape[1]),order='F')[...,np.newaxis]
            
            particles = val0[0,:,:]                     # Define the particles now
            particles_past = bt_MCMC[index_pf[-2],...]  # Define the particles after the last filter step
            delta_t = index_pf[-1] - index_pf[-2]       # Define the delta t as the number of integrations(IMPORTANT: In the case of real time assimilation the dt is variable.....)
            
            # Call particle filter 
            particles = particle_filter(ILC_a_cst,obs,K,Hpiv_Topos_K,particles,N_threshold,\
                                        np.concatenate((noises,noises_centered[np.newaxis,...]),axis=0)[index_pf[-2]:index_pf[-1],...],\
                                        particles_past,nb_mutation_steps,original_dt_simu,param['dt'],pho,delta_t,pchol_cov_noises) 
                                        
                                        
                                        
            
 
            val0 = particles[np.newaxis,...]                                         # Define the particles 
            param['dt'] = original_dt_simu                                           # Set the integration step as the original one 
            if (nb_assim ) == len(vector_of_assimilation_time)-1:                    # Set the next time of assimilation in the inf if there is no more data in the vecor
                next_time_of_assimilation = np.inf
            else:                                                                    # If there is data
                nb_assim += 1                                                        # Increments the Flag that controls the assimilation steps
                next_time_of_assimilation = vector_of_assimilation_time[nb_assim]    # Set the next time of assimilation
                
                
            assimilate_PIV = False      # Set the control Flag to False                                                                           

        ############################################################################################################
        #############################################################################################################
        if index==0:                                                                    # If the first time step integration
            noises = noises_centered[np.newaxis,...]                                    # Set the noises
        elif (next_time_of_assimilation != np.inf):                                     # If the next time of integration is not at inf, it saves the noise now
            noises = np.concatenate((noises,noises_centered[np.newaxis,...]),axis=0) 
            
            
        bt_MCMC = np.concatenate((bt_MCMC,val0),axis=0)    # Concatenate the particles in this time step with the particles before
        bt_fv   = np.concatenate((bt_fv,val1),axis=0)
        bt_m    = np.concatenate((bt_m,val2),axis=0)
        
        
        iii_realization = np.any(np.logical_or(np.isnan(bt_MCMC[index+1,:,:]),np.isinf(bt_MCMC[index+1,:,:])),axis = 0)[...,np.newaxis]  # Control if any realization has explosed
        
        
        if (time[-1]+param['dt'])>=(next_time_of_assimilation):  # If the next time integration will end after the time of assimilation, hence we need to change the time step 'dt' to end exactly in the same time of the observation
            param['dt'] = next_time_of_assimilation - time[-1]   # Therefore, the next time integration step will be the difference between the future and the present.
            assimilate_PIV = True                                # Set the Flag True
            
        
            ############################ Solve possible explosions in the integration
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
    
        
        ######################### Testing real time plot #######################
        
        if (index%20)==0 and (index!=0) and (plt_real_time==True):   # Plot at each 20 time steps 
            particles_mean = np.mean(bt_MCMC[:,:,:],axis=2)
            quantiles = np.quantile(bt_MCMC[:,:,:],q=[0.025,0.975],axis=2)
    
            ax_1.set_xlim([0, time[-1]+10])
            ax_2.set_xlim([0, time[-1]+10])
            ax_3.set_xlim([0, time[-1]+10])
            ax_4.set_xlim([0, time[-1]+10])
            ax_5.set_xlim([0, time[-1]+10])
            ax_6.set_xlim([0, time[-1]+10])
    
            
            
            line11.set_data(time,particles_mean[:,0])
            ax_1.collections.clear()
            ax_1.fill_between(time, quantiles[0,:,0],quantiles[1,:,0], color='gray')
            line13.set_data(np.array(time)[np.array(index_pf)[1:]],-2*np.ones((len(index_pf[1:]))))
            
            line21.set_data(time,particles_mean[:,1])
            ax_2.collections.clear()
            ax_2.fill_between(time, quantiles[0,:,1],quantiles[1,:,1], color='gray')
            line23.set_data(np.array(time)[np.array(index_pf)[1:]],-2*np.ones((len(index_pf[1:]))))
            
            line31.set_data(time,particles_mean[:,2])
#            ax_3.fill_between(time, quantiles[0,:,2],quantiles[1,:,2], color='gray')
            line33.set_data(np.array(time)[np.array(index_pf)[1:]],-2*np.ones((len(index_pf[1:]))))
            
            line41.set_data(time,particles_mean[:,3])
#            ax_4.fill_between(time, quantiles[0,:,3],quantiles[1,:,3], color='gray')
            line43.set_data(np.array(time)[np.array(index_pf)[1:]],-2*np.ones((len(index_pf[1:]))))
            
            line51.set_data(time,particles_mean[:,4])
#            ax_5.fill_between(time, quantiles[0,:,4],quantiles[1,:,4], color='gray')
            line53.set_data(np.array(time)[np.array(index_pf)[1:]],-2*np.ones((len(index_pf[1:]))))
            
            line61.set_data(time,particles_mean[:,5])
#            ax_6.fill_between(time, quantiles[0,:,5],quantiles[1,:,5], color='gray')
            line63.set_data(np.array(time)[np.array(index_pf)[1:]],-2*np.ones((len(index_pf[1:]))))
            
            fig.canvas.draw()
            plt.pause(0.005)    
    
    
    del bt_tronc
    
#    param['dt'] = param['dt']*n_simu
#    param['N_test'] = param['N_test']/n_simu
#    bt_MCMC = bt_MCMC[::n_simu,:,:]
#    bt_fv = bt_fv[::n_simu,:,:]
#    bt_m = bt_m[::n_simu,:,:]
#    bt_forecast_sto = bt_forecast_sto[::n_simu,:]
#    bt_forecast_deter = bt_forecast_deter[::n_simu,:]
    
#    struct_bt_MCMC = {}
#    
#    tot = {}
#    tot['mean'] = np.mean(bt_MCMC,2)
#    tot['var'] = np.var(bt_MCMC,2)
#    tot['one_realiz'] = bt_MCMC[:,:,0]
#    struct_bt_MCMC['tot'] = tot.copy()
#    
#    fv = {}
#    fv['mean'] = np.mean(bt_fv,2)
#    fv['var'] = np.var(bt_fv,2)
#    fv['one_realiz'] = bt_fv[:,:,0]
#    struct_bt_MCMC['fv'] = fv.copy()
#   
#    m = {}
#    m['mean'] = np.mean(bt_m,2)
#    m['var'] = np.var(bt_m,2)
#    m['one_realiz'] = bt_m[:,:,0]
#    struct_bt_MCMC['m'] = m.copy()
    
    
    #%%  Save 2nd results, especially I, L, C and the reconstructed Chronos

#    param = fct_name_2nd_result(param,modal_dt,reconstruction)
#    
#    
#    
##    np.savez(param['name_file_2nd_result']+'_Numpy',bt_forecast_deter=bt_forecast_deter,\
##                                                    bt_tot = bt_tot,\
##                                                    bt_forecast_sto = bt_forecast_sto,\
##                                                    param = param,\
##                                                    struct_bt_MCMC = struct_bt_MCMC,\
##                                                    bt_MCMC = bt_MCMC)  
##    
#    dict_python = {}
#    dict_python['bt_forecast_deter'] = bt_forecast_deter
#    dict_python['bt_tot'] = bt_tot
#    dict_python['bt_forecast_sto'] = bt_forecast_sto
#    dict_python['param'] = param
#    dict_python['struct_bt_MCMC'] = struct_bt_MCMC
#    dict_python['bt_MCMC'] = bt_MCMC
    
#    sio.savemat(param['name_file_2nd_result']+'_Numpy',dict_python)
    
    #%% PLOTSSSSSSSSSSSS
#    time_average = np.mean(bt_MCMC,axis=0)
#    var = np.sum(np.var(time_average,axis=1))
##    
    
    
    
    ##############################################################################################################
    #################################---TEST PLOTS---#############################################################
    if plt_real_time == False:
        dt_tot = param['dt']
        N_test = param['N_test'] 
    #    time = np.arange(1,int(int(N_test)+2),1)*float(dt_tot)
    #    ref = bt_MCMC[:,:,0]
        
        particles_mean = np.mean(bt_MCMC[:,:,:],axis=2)
        particles_median = np.median(bt_MCMC[:,:,:],axis=2)
        n_particles = bt_MCMC.shape[-1] 
    #    particles_std_estimate = np.std(bt_MCMC[:,:,1:],axis=2)
    #    erreur = np.abs(particles_mean-ref)
    
    
        quantiles = np.quantile(bt_MCMC[:,:,:],q=[0.025,0.975],axis=2)
      
        
    #    time_simu = np.arange(0,(bt_MCMC.shape[0])*dt_tot,dt_tot)
    #    time_bt_tot = np.arange(0,(bt_MCMC.shape[0])*dt_tot,n_simu*dt_tot*3)
    #    ref = bt_tot[:int(len(time_bt_tot)),:]
        for index in range(particles_mean.shape[1]):
            plt.figure(index)
            plt.ylim(-10, 10)
            ####        delta = 1.96*particles_std_estimate[:,index]/np.sqrt(n_particles)
            
            plt.fill_between(time,quantiles[0,:,index],quantiles[1,:,index],color='gray')
            line1 = plt.plot(time,particles_mean[:,index],'b-',label = 'Particles mean')
    #        line2 = plt.plot(time_bt_tot,ref[:,index],'k--',label = 'True state')
    #        line2 = plt.plot(time_bt_tot,ref[:,index],'k--',label = 'True state')
    #        line3 = plt.plot(time_simu,particles_median[:,index],'g.',label = 'particles median')
    #        line4 = plt.plot(dt_tot*np.concatenate((np.zeros((1)),np.array(time_pf))),particles_estimate[:,index],'m.',label = 'PF mean estimation')
            plt.plot(np.array(time)[np.array(index_pf)[1:]],-2*np.ones((len(index_pf[1:]))),'r.')
            plt.grid()
            plt.ylabel('Chronos '+r'$b'+str(index+1)+'$'+' amplitude',fontsize=20)
            plt.xlabel('Time',fontsize=20)
            plt.legend(fontsize=15)
        
        
        
        
#    '''
#    Now we have the estimation of the Chronos, therefore it's necessary to estabilish some measures to compare with the PIV measured flow. 
#    '''       
        
        
#    Champ_smoothed = Hpiv_Topos @ np.concatenate((particles_mean,np.ones((particles_mean.shape[0],1))),axis=1).T
    
#    topos_Fx = Hpiv_Topos[:14948,:]
#    topos_Fy = Hpiv_Topos[14948:,:]
#    
#    
#    delta_space = coordinates_x_PIV[1]-coordinates_x_PIV[0]
#    nb_x = topos_new_coordinates.shape[0]
#    nb_y = topos_new_coordinates.shape[1]
#    curl_Z = calculate_rotational(topos_Fx,topos_Fy,delta_space,nb_x,nb_y)
#    
#    Champ_smoothed = np.reshape(curl_Z,(int(curl_Z.shape[0]*curl_Z.shape[1]),curl_Z.shape[2]),order='F') @ np.concatenate((particles_mean,np.ones((particles_mean.shape[0],1))),axis=1).T
#    
#    Champ_smoothed_reshaped = np.reshape(Champ_smoothed,(nb_x-2,nb_y-2,Champ_smoothed.shape[-1]),order='F')   
#    
#    
#    for i in range(Champ_smoothed_reshaped.shape[-1]):
#        plt.imshow(Champ_smoothed_reshaped[:,:,i].T)
#        
#        plt.pause(0.1)
#       
#    
#    PIV_Fx = matrix_data_PIV_all_data[:,:,0].T
#    PIV_Fy = matrix_data_PIV_all_data[:,:,1].T
#    curl_Z_PIV = calculate_rotational(PIV_Fx,PIV_Fy,delta_space,nb_x,nb_y)
    
#    for i in range(curl_Z_PIV.shape[-1]):
#        plt.imshow(curl_Z_PIV[:,:,i].T)
#        plt.pause(0.001)
    
    
        

#################################   SAVE 
#    
#    dict_python = {}
#    dict_python['particles_mean'] = particles_mean[time_pf,:]
#    dict_python['MX'] = param['MX']
##    dict_python['index_filter'] = time_pf
#        
#    name_file_data = Path(__file__).parents[3].joinpath('data_after_filtering').joinpath(str(particles_mean.shape[1])+'modes_particle_mean')
#    sio.savemat(str(name_file_data)+'_Numpy',dict_python)
    ##############################################################################################################
    ##############################################################################################################
#    dict_python = {}
#    dict_python['obs'] = observations_to_save
#    dict_python['bt_MCMC'] = particles_mean
#    dict_python['particles'] = bt_MCMC
#    dict_python['iters'] = param['N_test']
#    dict_python['dt'] = param['dt']
#    dict_python['bt_tot'] = bt_tot
#    dict_python['n_simu'] = n_simu
#    dict_python['param'] = param
#    dict_python['index_of_filtering'] = index_of_filtering
#    name_file_data = Path(__file__).parents[3].joinpath('test').joinpath('data_to_benchmark'+str(nb_modes)+'_bruit_'+str(beta_1)+'reynolds300')
#    sio.savemat(str(name_file_data),dict_python)
    
    
    
    
    del C_deter 
    del C_sto 
    del L_deter 
    del L_sto 
    del I_deter 
    del I_sto

    
    
    
    return 0 #var
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    