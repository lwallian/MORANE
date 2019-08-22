# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:58:23 2019

@author: matheus.ladvig
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.linear_model import LinearRegression


path_data = 'C:\\Users\\matheus.ladvig\\Desktop\\mecanique de fluides\\spectres\\spectres\\cdm\\dsp_piv_bulles'
path_to_file = Path(path_data)
path_to_file_piv = path_to_file.joinpath('dsp_-003dw_123.txt')
print(path_to_file_piv)


path_data = 'C:\\Users\\matheus.ladvig\\Desktop\\mecanique de fluides\\spectres\\spectres\\cdm\\dsp_fil_chaud'
path_to_file = Path(path_data)
path_to_file_fil = path_to_file.joinpath('dsp_bulles_-003dw.txt')
print(path_to_file_fil)




#################### Noise determination in direction u ########################


u_inf = 1.91
D = 32*10**(-3)
U_a = 1.5
U_b = 1
delta_U = U_a - U_b
fp = 5.13
const = (delta_U**2)/fp





dsp_fil_u = (np.loadtxt(path_to_file_fil)[:, 1])/const
dsp_fil_v = (np.loadtxt(path_to_file_fil)[:, 2])/const
freq_fil  = (np.loadtxt(path_to_file_fil)[:, 0])
#freq_fil_normalized = freq_fil/fp
#delta_f_fil = freq_fil[2]-freq_fil[1]


dsp_piv_u_1 = np.loadtxt(path_to_file_piv)[:, 1]
dsp_piv_u_2 = np.loadtxt(path_to_file_piv)[:, 4]
dsp_piv_u_3 = np.loadtxt(path_to_file_piv)[:, 7]
dsp_piv_u = np.mean(np.vstack((dsp_piv_u_1,dsp_piv_u_2,dsp_piv_u_3)),axis=0)/const
freq_piv  = (np.loadtxt(path_to_file_piv)[:, 0])
#freq_piv_normalized = freq_piv/fp
#delta_f_piv = freq_piv[2]-freq_piv[1]
#
#
dsp_piv_v_1 = np.loadtxt(path_to_file_piv)[:, 2]
dsp_piv_v_2 = np.loadtxt(path_to_file_piv)[:, 5]
dsp_piv_v_3 = np.loadtxt(path_to_file_piv)[:, 8]
dsp_piv_v = np.mean(np.vstack((dsp_piv_v_1,dsp_piv_v_2,dsp_piv_v_3)),axis=0)/const
#
#

indexes_freq_noise = np.where((freq_piv>50))[0]
freq_noise = freq_piv[indexes_freq_noise]
amplitude_energy_noise = dsp_piv_u[indexes_freq_noise]
mean_noise = np.mean(amplitude_energy_noise)



plt.figure()
plt.loglog(freq_fil,dsp_fil_u,'b--',label = 'Fil chaud')
plt.loglog(freq_piv,dsp_piv_u,'g--',label = 'PIV')
plt.loglog(freq_noise,amplitude_energy_noise,'y',label = 'Noise')
plt.loglog(freq_noise,mean_noise*np.ones(len(freq_noise)),'r',label = 'Average noise')
plt.grid(True,which="both",ls="-")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Energy(J)')
plt.legend()


plt.figure()
plt.loglog(freq_fil,dsp_fil_v)
plt.loglog(freq_piv,dsp_piv_v)
plt.grid(True,which="both",ls="-")

########################################### Noise estimation ###########################################


noise_variance= mean_noise
ecartype_noise = np.sqrt(noise_variance)

ecartype_noise_meters = 1.295*ecartype_noise
########################################### Filter estimation ###########################################
indexes_freq_piv = np.where((freq_piv<50))[0]
indexes_freq_fil = np.where((freq_fil<50))[0]

freq_filter_piv = freq_piv[indexes_freq_piv]
freq_filter_fil = freq_fil[indexes_freq_fil]

signal_filtered = dsp_piv_u[indexes_freq_piv]
signal_original = dsp_fil_u[indexes_freq_fil]


plt.figure()
plt.loglog(freq_filter_piv,signal_filtered,label = 'Signal Filtered')
plt.loglog(freq_filter_fil,signal_original,label = 'Signal')
plt.grid(True,which="both",ls="-")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Energy(J)')
plt.legend()





#  Interpolation
tck = interpolate.splrep(freq_filter_fil, signal_original, s=0)
signal_original_interpo = interpolate.splev(freq_filter_piv, tck, der=0)


plt.figure()
plt.loglog(freq_filter_fil,signal_original,'o-',label = 'Signal sampled')
plt.loglog(freq_filter_piv,signal_original_interpo,'.-',label = 'Signal interpolated to simulate another sampling frequency')
plt.grid(True,which="both",ls="-")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Energy(J)')
plt.legend()


# Filter H estimation

H_filter = np.sqrt(signal_filtered/signal_original_interpo)

H_filter = H_filter/H_filter[0]

plt.figure()
plt.plot(freq_filter_piv,H_filter,label = 'Filter frequency response')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Find gaussian curve over the H Filter

x = np.power(freq_filter_piv,2)
y = np.log(H_filter)

model = LinearRegression().fit(x[...,np.newaxis], y[...,np.newaxis])

b0 = model.intercept_
b1 = model.coef_


data = np.exp(b1[0,0]*x)

plt.figure()
plt.plot(freq_filter_piv,H_filter,label = 'Filter frequency response')
plt.plot(freq_filter_piv,data,label = 'Equivalent Gaussian filter frequency response')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()



#################### following the fourier transform in time domain, we have the std deviation as

#var = -b1[0,0]/(2*(0.75**2)*(np.pi**2))
#std = np.sqrt(var)
#print('Standart deviation: '+str(std))
#
#t=np.arange(0,5*std,std/10)
##np.sqrt(-0.75*np.pi/b1)
#filter_time = np.sqrt(-0.75*np.pi/b1)*np.exp(-t**2/(2*var))




std_in_sec = np.sqrt(-b1[0,0]/(2*np.pi**2))

std_in_meters = (1.295*std_in_sec)



















































