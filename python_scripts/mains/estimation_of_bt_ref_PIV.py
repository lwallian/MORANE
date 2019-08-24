# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:28:29 2019

@author: matheus.ladvig
"""

#
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
import hdf5storage
import os
from convert_mat_to_python import convert_mat_to_python
#from pathlib import Path

Plot_error_bar = False

#                           DATASET 
#    type_data = 'incompact3D_noisy2D_40dt_subsampl_truncated'  #dataset to debug
type_data = 'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated' # Reynolds 300
#    type_data = 'DNS100_inc3d_2D_2018_11_16_blocks_truncated'  # Reynolds 100
switcher = {
    'incompact3D_noisy2D_40dt_subsampl_truncated': [[1e-05],[False]],
    'DNS100_inc3d_2D_2018_11_16_blocks_truncated': [[1e-06],[True]], # BEST
    'turb2D_blocks_truncated': [[1e-05],[False,True]],
    'incompact3d_wake_episode3_cut_truncated': [[1e-06],[False]],
    'incompact3d_wake_episode3_cut': [[1e-06],[False]],
    'LES_3D_tot_sub_sample_blurred': [[1e-03],[True]],
    'inc3D_Re3900_blocks': [[1e-03],[True]],
    'inc3D_Re3900_blocks_truncated': [[1e-03],[True]],
    'DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated':[[1e-04],[True]] # BEST
   
}
#switcher.get(argument,[[0.0005],[False]])
# Get threshold and modal_dt
threshold,modal_dt = switcher.get(type_data)
threshold=threshold[0]
modal_dt=modal_dt[0]
no_subampl_in_forecast = False 
plot_debug = False
data_assimilate_dim = 2
nb_modes = 6
dt_PIV = 0.080833
Re = 300
test_fct = 'b'
#svd_pchol = True
choice_n_subsample = 'auto_shanon'
adv_corrected = [False]


current_pwd = Path(__file__).parents[1] # Select the path
folder_results = current_pwd.parents[0].joinpath('resultats').joinpath('current_results') # Select the current results path
folder_data = current_pwd.parents[0].joinpath('data') # Select the data path
#param_ref['folder_results'] = str(folder_results) # Stock folder results path
#param_ref['folder_data'] = str(folder_data)       # Stock folder data path
#modal_dt_ref = modal_dt # Define modal_dt_ref
 
    
    #%% Get data
    
# On which function the Shanon ctriterion is used
#    test_fct = 'b'  # 'b' is better than db
a_t = '_a_cst_' 

############################ Construct the path to select the model constants I,L,C,pchol and etc.

file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
        a_t + '_decor_by_subsampl_bt_decor_choice_' + choice_n_subsample + \
        '_threshold_' + str(threshold) + \
        'fct_test_' + test_fct    
#    file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
#            a_t + '_decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_' + str(threshold) + \
#            'fct_test_' + test_fct    
#    var_exits =  'var' in locals() or 'var' in globals()
#    period_estim = 'period_estim' in locals() or 'period_estim' in globals()
#    if var_exits == True and period_estim == True:
#        file = file + '_p_estim_' + str(period_estim);
file = file + '_fullsto' # File where the ROM coefficients are save
print(file)
#    file_save = file
if not adv_corrected:
    file = file + '_no_correct_drift'
file = file + '.mat'
file_res = folder_results / Path(file)
if not os.path.exists(file_res):
    file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_'  \
              + choice_n_subsample
    if choice_n_subsample == 'auto_shanon' :
        file = file + '_threshold_' + str(threshold)
    file = file + test_fct   
    file = file + '_fullsto' # File where the ROM coefficients are save
#        file = '1stresult_' + type_data + '_' + str(nb_modes) + '_modes_' + \
#            choice_n_subsample + '_threshold_' + str(threshold) + test_fct   
#        file = file_save
    if not adv_corrected:
        file = file + '\\_no_correct_drift'
    file = file + '.mat'
    file_res = folder_results / Path(file)
print(file)

# The function creates a dictionary with the same structure as the Matlab Struct in the path file_res
I_sto,L_sto,C_sto,I_deter,L_deter,C_deter,plot_bts,pchol_cov_noises,bt_tot,param = convert_mat_to_python(str(file_res)) # Call the function and load the matlab data calculated before in matlab scripts.
param['decor_by_subsampl']['no_subampl_in_forecast'] = no_subampl_in_forecast                                           # Define the constant
    


'''
Load HpivTopos and PIV
'''
#Hpiv_Topos = np.load('Hpiv_Topos.npy')

#      LOAD TOPOS
print('Loading H_PIV @ Topos...')
path_topos = Path(folder_data).parents[1].joinpath('data_PIV').\
        joinpath('mode_'+type_data+'_'+str(nb_modes)+'_modes_PIV') # Topos path 
topos_data = hdf5storage.loadmat(str(path_topos))                                                                 # Load topos
topos = topos_data['phi_m_U']   
MX_PIV = topos_data['MX_PIV'].astype(int)
MX_PIV = tuple(map(tuple,MX_PIV))[0]
coordinates_x_PIV= topos_data['x_PIV_after_crop']
coordinates_y_PIV= topos_data['y_PIV_after_crop']
coordinates_x_PIV= np.reshape(coordinates_x_PIV,MX_PIV,order='F') 
coordinates_y_PIV= np.reshape(coordinates_y_PIV,MX_PIV,order='F') 
coordinates_x_PIV =  np.transpose(coordinates_x_PIV[:,0])  
coordinates_y_PIV = coordinates_y_PIV[0,:]   
topos_new_coordinates = np.reshape(topos,\
                          MX_PIV + tuple(np.array([data_assimilate_dim,(nb_modes+1)])),order='F') 
Hpiv_Topos = np.reshape(topos_new_coordinates,(int(topos_new_coordinates.shape[0]*topos_new_coordinates.shape[1]*topos_new_coordinates.shape[2]),topos_new_coordinates.shape[3]),order='F') # The topos that we have estimated reshaped to posterior matrix multiplications



#%%   Calculate Sigma for LS variance estimation
if Plot_error_bar:
    print('Loading Sigma')
    threshold_ = str(threshold).replace('.', '_',)
    path_Sigma_inverse = Path(__file__).parents[3].joinpath('data_PIV').\
    joinpath('HSigSigH_PIV_'+type_data+'_'+str(param['nb_modes'])\
             +'_modes_a_cst_threshold_'+ threshold_)  # Load Sigma_inverse
    #    path_Sigma_inverse = Path(__file__).parents[3].joinpath('data_PIV').joinpath('HSigSigH_PIV_'+type_data+'_'+str(param['nb_modes'])+'_modes_a_cst_threshold_0_'+str(threshold)[2:])  # Load Sigma_inverse
    Sigma_data = hdf5storage.loadmat(str(path_Sigma_inverse)) # Select Sigma_inverse
    Sigma = Sigma_data['HSigSigH'][:,0,:,:]                             # Load Sigma inverse 
    ##### Transform this matrix in a square matrix
    nb_points = Sigma.shape[0]                                                      # Number of points in the grid
    nb_dim = Sigma.shape[1]            



'''
Load PIV
'''
path_y = Path(folder_data).parents[1].joinpath('data_PIV').\
        joinpath('Data_piv.npy')
y = np.load(path_y)[:4082,:]

'''
Reshape Piv data in order to compare with topos
'''
y = np.reshape(y,(y.shape[0],int(y.shape[1]*y.shape[2])),order='F')


'''
get error from PIV with average velocity
'''
y_less_average = y - Hpiv_Topos[...,-1]


'''
Get the topos with chronos different from 1

n -> n-1
'''
Hpiv_Topos_otimization = Hpiv_Topos[:,:-1].copy()



'''
Find the best chronos in all piv data in this inverse problem.
'''
valeurs = np.zeros((y_less_average.shape[0],Hpiv_Topos_otimization.shape[1]))
for time in range(y.shape[0]):
    print(time)
    reg = linear_model.RidgeCV(alphas=np.logspace(-2.5, 2.5, 30))
    reg.fit(Hpiv_Topos_otimization,y[time:time+1,...].T)       
    valeurs[time,:] = reg.coef_


#%%   Calculate Sigma for LS variance estimation
if Plot_error_bar:
    pinv_Hpiv = np.linalg.pinv(Hpiv_Topos_otimization)
    cov = np.zeros((int(nb_modes),int(nb_modes+1)))
    # Calculating necessary matrices
    #K = np.zeros((int(nb_dim),int(nb_modes+1),int(nb_points)))
    Sigma = np.transpose(Sigma,(1,2,0))
    pinv_Hpiv = np.reshape(pinv_Hpiv,(int(nb_modes),int(nb_points),int(nb_dim)),order='F') 
    pinv_Hpiv = np.transpose(pinv_Hpiv,(0,2,1))
    for line in range(int(nb_points)):                                                  # To all spatial samples we create the first part of the matrix that contains the correlation of Vx
        cov = cov + \
            pinv_Hpiv[:,:,line] @ Sigma[:,:,line] @ pinv_Hpiv[:,:,line].T
    #K = np.transpose(K,(2,0,1)) # ((nb_points),(nb_dim),(nb_modes+1),)
    #K = np.reshape(K,(int(nb_points*nb_dim),int(nb_modes+1)),order='F') 
    #Hpiv_Topos = np.transpose(Hpiv_Topos,(2,0,1)) # ((nb_points),(nb_dim),(nb_modes+1),)
    #Hpiv_Topos = np.reshape(Hpiv_Topos,(int(nb_points*nb_dim),int(nb_modes+1)),order='F') 
    #Sigma_inverse = np.transpose(Sigma_inverse,(2,0,1))  # ((nb_points),(nb_dim),(nb_dim),)
    estim_err = 1.96*sqrt(np.diag(cov))


'''
Plot the result for the chronos found
'''
t = np.arange(0,valeurs.shape[0]*dt_PIV,dt_PIV)
for i in range(valeurs.shape[1]):
    plt.figure()
    plt.plot(t,valeurs[:,i])



'''
Compare the error in the time average PIV flow with the average topos(-> Hpiv[-1])
'''
average_time_value_PIV = np.mean(y,axis=0)
error = Hpiv_Topos[:,-1] - average_time_value_PIV
error_reshaped = np.reshape(error,(202,74,2),order='F')

# PLOT
plt.figure()
plt.imshow(error_reshaped[:,:,0].T)
plt.colorbar()
plt.figure()
plt.imshow(error_reshaped[:,:,1].T)
plt.colorbar()


#Plot of the average flow for both
plt.figure()
plt.imshow((np.reshape(Hpiv_Topos[:,-1],(202,74,2),order='F')[:,:,0]).T)
plt.colorbar()
plt.figure()
plt.imshow((np.reshape(average_time_value_PIV,(202,74,2),order='F')[:,:,0]).T)
plt.colorbar()



'''
The Chronos found here need to be saved and will be necessary to evaluate the particle filterning
'''


dict_python = {}
dict_python['Hpiv_Topos_x'] = np.reshape(Hpiv_Topos[:,-1],(202,74,2),order='F')[:,:,0]
dict_python['Hpiv_Topos_y'] = np.reshape(Hpiv_Topos[:,-1],(202,74,2),order='F')[:,:,1]
dict_python['average_time_value_PIV_x'] = np.reshape(average_time_value_PIV,(202,74,2),order='F')[:,:,0]
dict_python['average_time_value_PIV_y'] = np.reshape(average_time_value_PIV,(202,74,2),order='F')[:,:,1]
dict_python['bt_tot_PIV'] = valeurs
dict_python['dt_PIV'] = dt_PIV
dict_python['Re'] = Re

file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('bt_tot_PIV_Re'+str(dict_python['Re'])+'.mat')
#data = hdf5storage.loadmat(str(file))
sio.savemat(file,dict_python)




















