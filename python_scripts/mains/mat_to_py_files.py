# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:09:54 2019

@author: matheus.ladvig
"""

PATH_MAT_FILE = 'D:/python_scripts/resultats/current_results/1stresult_incompact3D_noisy2D_40dt_subsampl_truncated_2_modes__a_cst__decor_by_subsampl_bt_decor_choice_auto_shanon_threshold_1e-05fct_test_b_fullsto_no_correct_drift.mat'



import hdf5storage # import the package resposible to convert

# Load the mat file of this path
mat = hdf5storage.loadmat(PATH_MAT_FILE)


# The param has the key 'param'
param = mat['param']
# Create the dictionary that will stock the data
param_dict = {}

# Loop to run all the dtypes in the data
for name in param.dtype.names :
    
    
    # If the dtype has only one information we take his name and create a key for this information
    if param[0][name].dtype.names == None:
        if param[0][name].shape == (1,1):
            param_dict[name] = param[0][name][0,0]
        else:
            param_dict[name] = param[0][name]
    # If not, we need to create a dict that will be resposnsible to stock all fields of this structure
    
    else:
        aux_dict = {}
        for name2 in param[0][name].dtype.names:
            if param[0][name][name2].shape == (1,1,1):
                aux_dict[name2] = param[0][name][name2][0,0,0]
            else:
                aux_dict[name2] = param[0][name][name2]
        
        param_dict[name] = aux_dict
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    