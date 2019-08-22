# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:03:41 2019

@author: matheus.ladvig
"""
import hdf5storage
import numpy as np
from pathlib import Path

path_load_matlab =  Path(__file__).parents[3].joinpath('test').joinpath('savematlab')
mat = hdf5storage.loadmat(str(path_load_matlab))
mat_matrix = mat['var_save']


path_load_python =  Path(__file__).parents[3].joinpath('test').joinpath('pythonfile.npy')
npy_matrix = np.load(path_load_python)


erreur = np.sqrt(np.power(npy_matrix - mat_matrix.T,2)).T

