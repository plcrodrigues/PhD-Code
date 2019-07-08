#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:07:52 2018

@author: coelhorp

This is an example with data from the GigaDB dataset, which is publicly
available at http://gigadb.org/dataset/100295

The original dataset contains recordings from experiments on 52 subjects
performing motor imagery tasks for two classes and with 64 electrodes.
The experiments consisted of 5 runs with 40 trials, making a total of 200
trials in the dataset (100 for each class).

We included in the /datasets folder of this repository the data
from subjects 1, 26, and 43 (here they are called subjects 1, 2, and 3, respectively).

We followed the classic preprocessing pipeline for BCI signals in motor imagery,
applying a 8-35 Hz bandpass filter to the signals and then taking epochs
of 2 seconds for each experimental trial. We downsampled the electrodes to
22 channels, discarding the electrodes from regions that we knew were not
physiologically relevant for the cognitive task performed in this experiment.
Finally, we estimated the spatial covariance matrices for each of the trials
and stored them in a pickle (.pkl) file accessible from the /datasets folder.
Note that the .pkl file was saved in Python 3, so you will probably have
problems to open it with Python 2.

Necessary packages to make this script work (all downloadable via pip) :
	•	pyRiemann
	•	numpy
	•	scikit-learn

"""

import numpy as np
from pyriemann.utils.mean import mean_riemann, mean_euclid, mean_harmonic
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import powm, invsqrtm, sqrtm
from pyriemann.estimation import Covariances
from riemann_lab import get_datasets as GD

def power_means(C, p):
    phi = 0.375/np.abs(p)
    K = len(C)
    n = C[0].shape[0]
    w = np.ones(K)
    w = w/(1.0*len(w))
    G = np.sum([wk*powm(Ck, p) for (wk,Ck) in zip(w,C)], axis=0)
    if p > 0:
        X = invsqrtm(G)
    else:
        X = sqrtm(G)
    zeta = 10e-10
    test = 10*zeta
    while test > zeta:
        H = np.sum([wk*powm(np.dot(X, np.dot(powm(Ck, np.sign(p)), X.T)), np.abs(p)) for (wk,Ck) in zip(w,C)], axis=0)
        X = np.dot(powm(H, -phi), X)
        test = 1.0/np.sqrt(n) * np.linalg.norm(H - np.eye(n))
    if p > 0:
        P = np.dot(np.linalg.inv(X), np.linalg.inv(X.T))
    else:
        P = np.dot(X.T, X)
    return P

from sklearn.externals import joblib

dataset = 'Cho2017'
storage = 'GIPSA'
settings = GD.get_settings(dataset, storage)
session = settings['session']

subject = 1
data = GD.get_dataset(dataset, subject, session, storage)
covs = Covariances(estimator='oas').fit_transform(data['signals'])
labs = data['labels']

print('euclidean mean')
M_euclid = mean_euclid(covs)
M_pmeans = power_means(covs, p=+1)
print('difference:', distance_riemann(M_euclid, M_pmeans))
print('')

print('harmonic mean')
M_harmonic = mean_harmonic(covs)
M_pmeans = power_means(covs, p=-1)
print('difference:', distance_riemann(M_harmonic, M_pmeans))
print('')

print('riemannian mean')
M_riemann = mean_riemann(covs)
M_pmeans_pos = power_means(covs, p=+0.001)
M_pmeans_neg = power_means(covs, p=-0.001)
M_pmeans = geodesic_riemann(M_pmeans_pos, M_pmeans_neg, alpha=0.5)
print('difference:', distance_riemann(M_riemann, M_pmeans))
print('')
