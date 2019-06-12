# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:58:23 2019

@author: matheus.ladvig
"""

import numpy
from pathlib import Path


path_data = 'C:\\Users\\matheus.ladvig\\Desktop\\mecanique de fluides\\spectres\\spectres\\cylinder_wake\\dsp_piv_bulles'
path_to_file = Path(path_data)
path_to_file = path_to_file.joinpath('dsp_3D_12.txt')
print(path_to_file)

file_handle = open(path_to_file, 'r')

lines_list = file_handle.readlines()


