#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:33:59 2017

@author: coelhorp
"""

from collections import OrderedDict
import numpy as np

from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.tangentspace import TangentSpace, tangent_space, untangent_space
from pyriemann.estimation import Covariances, XdawnCovariances, ERPCovariances
from pyriemann.utils.mean import mean_riemann, geodesic_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from pyriemann.utils.distance import distance_riemann
from pyriemann.classification import MDM

from sklearn.externals import joblib
from . import manopt as manifoptim 

def get_sourcetarget_split_motorimagery(source, target, ncovs_train):

    nclasses = len(np.unique(target['labels']))

    if not(hasattr(ncovs_train, "__len__")):
        ncovs_train = [ncovs_train] * nclasses

    target_train_idx = []
    for j, label in enumerate(np.unique(target['labels'])):
        sel = np.arange(np.sum(target['labels'] == label))
        np.random.shuffle(sel)
        target_train_idx.append(np.arange(len(target['labels']))[target['labels'] == label][sel[:ncovs_train[j]]])
    target_train_idx = np.concatenate(target_train_idx)
    target_test_idx  = np.array([i for i in range(len(target['labels'])) if i not in target_train_idx])

    target_train = {}
    target_train['covs'] = target['covs'][target_train_idx]
    target_train['labels'] = target['labels'][target_train_idx]

    target_test = {}
    target_test['covs'] = target['covs'][target_test_idx]
    target_test['labels'] = target['labels'][target_test_idx]    

    return source, target_train, target_test

def get_sourcetarget_split_p300(source, target, ncovs_train):

    X_source = source['epochs']
    y_source = source['labels'].flatten()
    covs_source = ERPCovariances(classes=[2], estimator='lwf').fit_transform(X_source, y_source)

    source = {}
    source['covs'] = covs_source
    source['labels'] = y_source

    X_target = target['epochs']
    y_target = target['labels'].flatten()

    sel = np.arange(len(y_target))
    np.random.shuffle(sel)
    X_target = X_target[sel]
    y_target = y_target[sel]

    idx_erps = np.where(y_target == 2)[0][:ncovs_train]
    idx_rest = np.where(y_target == 1)[0][:ncovs_train*5] # because there's one ERP in every 6 flashes

    idx_train = np.concatenate([idx_erps, idx_rest])
    idx_test  = np.array([i for i in range(len(y_target)) if i not in idx_train])

    erp = ERPCovariances(classes=[2], estimator='lwf')
    erp.fit(X_target[idx_train], y_target[idx_train])

    target_train = {}
    covs_target_train = erp.transform(X_target[idx_train])
    y_target_train = y_target[idx_train]
    target_train['covs'] = covs_target_train
    target_train['labels'] = y_target_train

    target_test = {}
    covs_target_test = erp.transform(X_target[idx_test])
    y_target_test = y_target[idx_test]
    target_test['covs'] = covs_target_test
    target_test['labels'] = y_target_test

    return source, target_train, target_test

def parallel_transport_covariance_matrix(C, R):
    return np.dot(invsqrtm(R), np.dot(C, invsqrtm(R)))

def parallel_transport_covariances(C, R):
    Cprt = []
    for Ci, Ri in zip(C, R):
        Cprt.append(parallel_transport_covariance_matrix(Ci, Ri))
    return np.stack(Cprt)

def transform_org2rct(source, target_train, target_test, weights_classes=None):

    weights_source = np.ones(len(source['labels']))
    weights_target = np.ones(len(target_train['labels']))
    if weights_classes is not None:
        for label in weights_classes.keys():
            weights_source[source['labels'] == label] = weights_classes[label]
            weights_target[target_train['labels'] == label] = weights_classes[label]

    source_rct = {}
    source_rct['labels'] = source['labels']
    T = mean_riemann(source['covs'], sample_weight=weights_source)
    T_source = np.stack([T]*len(source['covs']))
    source_rct['covs'] = parallel_transport_covariances(source['covs'], T_source)

    target_rct_train = {}
    target_rct_train['labels'] = target_train['labels']
    M_train = mean_riemann(target_train['covs'], sample_weight=weights_target)
    T_target = np.stack([M_train]*len(target_train['covs']))
    target_rct_train['covs'] = parallel_transport_covariances(target_train['covs'], T_target)

    target_rct_test = {}
    target_rct_test['labels'] = target_test['labels']
    M_test = M_train
    T_target = np.stack([M_test]*len(target_test['covs']))
    target_rct_test['covs'] = parallel_transport_covariances(target_test['covs'], T_target)

    return source_rct, target_rct_train, target_rct_test

def transform_rct2str(source, target_train, target_test):

    covs_source = source['covs']
    covs_target_train = target_train['covs']
    covs_target_test = target_test['covs']

    source_pow  = {}
    source_pow ['covs'] = source['covs']
    source_pow ['labels'] = source['labels']

    n = covs_source.shape[1]
    disp_source = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_source]) / len(covs_source)
    disp_target = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_target_train]) / len(covs_target_train)
    p = np.sqrt(disp_target / disp_source)

    target_pow_train  = {}
    target_pow_train['covs'] = np.stack([powm(covi, 1.0/p) for covi in covs_target_train])
    target_pow_train['labels'] = target_train['labels']

    target_pow_test  = {}
    target_pow_test['covs'] = np.stack([powm(covi, 1.0/p) for covi in covs_target_test])
    target_pow_test['labels'] = target_test['labels']

    return source_pow , target_pow_train, target_pow_test

def transform_str2rot(source, target_train, target_test, weights_classes=None, distance='euc'):

    source_rot = {}
    source_rot['covs'] = source['covs']
    source_rot['labels'] = source['labels']

    target_rot_train = {}
    target_rot_train['labels'] = target_train['labels']

    target_rot_test = {}
    target_rot_test['labels'] = target_test['labels']

    class_labels = np.unique(source['labels'])

    M_source = []
    for i in class_labels:
        M_source_i = mean_riemann(source['covs'][source['labels'] == i])
        M_source.append(M_source_i)

    M_target_train = []
    for j in class_labels:
        M_target_train_j = mean_riemann(target_train['covs'][target_train['labels'] == j])
        M_target_train.append(M_target_train_j)

    if weights_classes is None:
        weights = [1] * len(class_labels)
    else:
        weights = []
        for label in class_labels:
            weights.append(weights_classes[label])

    R = manifoptim.get_rotation_matrix(M=M_source, Mtilde=M_target_train, dist=distance, weights=weights)    

    covs_target_train = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_train['covs']])
    target_rot_train['covs'] = covs_target_train

    covs_target_test = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_test['covs']])
    target_rot_test['covs'] = covs_target_test

    return source_rot, target_rot_train, target_rot_test


