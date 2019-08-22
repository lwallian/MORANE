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



'''
Load HpivTopos and PIV
'''
Hpiv_Topos = np.load('Hpiv_Topos.npy')
y = np.load('Data_piv.npy')[:4082,:]

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



'''
Plot the result for the chronos found
'''
t = np.arange(0,valeurs.shape[0]*0.080833,0.080833)
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
dict_python['dt_PIV'] = 0.080833
dict_python['Re'] = 300

file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('bt_tot_PIV_Re'+str(dict_python['Re'])+'.mat')
#data = hdf5storage.loadmat(str(file))
sio.savemat(file,dict_python)




















