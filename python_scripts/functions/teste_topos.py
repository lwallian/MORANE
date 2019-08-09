# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:48:07 2019

@author: matheus.ladvig
"""
from pathlib import Path
import hdf5storage



name = 'mode_DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated_6_modes'

path_topos = (Path(__file__).parents[3]).joinpath('data').joinpath(name)
print(path_topos)

mat = hdf5storage.loadmat(str(path_topos))





