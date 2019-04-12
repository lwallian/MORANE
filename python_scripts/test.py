# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:09:31 2019

@author: matheus.ladvig
"""

import numpy as np



weights = np.array([0.4,0.05,0.1,0.14,0.31])

n = len(weights)
indices = []
C = [0.0] + [np.sum(weights[: i + 1]) for i in range(n)]
u0, j = np.random.random(), 0
for u in [(u0 + i) / n for i in range(n)]:
    while u > C[j]:
        j += 1
    indices.append(j - 1)
    
    