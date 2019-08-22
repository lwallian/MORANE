# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:06:54 2019

@author: matheus.ladvig
"""

import numpy as np


a = np.array([[1,2,3,4,5,6,7]])

chronos = np.tile(a,(60,1))


topos = np.ones((7,400,3))

#################################################################
e = np.transpose(np.tile(chronos,(400,3,1,1)),(2,3,0,1))

topos = np.tile(topos,(60,1,1,1))


vector = np.multiply(e,topos)

vector = np.sum(vector,axis=1)

