#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:08:03 2018

@author: coelhorp
"""

import numpy as np
from pyriemann.utils.distance import distance_riemann, distance_euclid
from pyriemann.utils.mean import mean_riemann

def gen_spd_rangfaible(n, eps, eigsmin=1):
    M = np.random.randn(n,n)
    M = M + M.T
    _,Q = np.linalg.eig(M)
    w = 1 + np.random.rand(n)
    w[-1] = eps * np.sqrt(np.sum(w[:-1]**2))
    w = np.diag(w)
    M = Q @ w @ Q.T
    return M

n = 3
p = 2

nrzt = 10
X = np.stack([gen_spd_rangfaible(n, eps=0.01) for _ in range(nrzt)], axis=0)
M = mean_riemann(X)
for Xi in X:
    print(distance_riemann(Xi, M))
    print(distance_euclid(Xi, M))    
    print('')