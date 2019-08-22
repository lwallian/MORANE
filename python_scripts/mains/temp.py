# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio


current_pwd = Path(__file__).parents[1].joinpath('testeeu')


#join = Path('realpython\eu.txt')

#a = np.array([1,2,3,4])[...,np.newaxis,np.newaxis]
#b = np.array([1,200,3,4])
#d = {}
#d['r'] = np.array([1,2])
#c = {}
###
#c['a'] = a
#c['b'] = b
#c['d'] = d
#sio.savemat(str(current_pwd),c)


data = sio.loadmat(str(current_pwd))

