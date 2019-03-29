# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:24:21 2019

@author: matheus.ladvig
"""

import hdf5storage # import the package resposible to convert
mat = hdf5storage.loadmat('d:/python_scripts/data/incompact3D_noisy2D_40dt_subsampl_truncated_2_modes_threshold_1e-05_nb_period_test_NaN_Chronos_test_basis.mat')

bt = mat['bt']
truncated_error2 = mat['truncated_error2']
param['']