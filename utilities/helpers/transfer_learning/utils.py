#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:33:59 2017

@author: coelhorp
"""

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, powm
from pyriemann.utils.distance import distance_riemann
from . import manopt as manifoptim

from pyriemann.transfer import (TLCenter, TLStretch,
                                TLRotate, encode_domains)


# def get_sourcetarget_split_motorimagery(source, target, ncovs_train):

#     nclasses = len(np.unique(target['labels']))

#     if not(hasattr(ncovs_train, "__len__")):
#         ncovs_train = [ncovs_train] * nclasses

#     target_train_idx = []
#     for j, label in enumerate(np.unique(target['labels'])):
#         sel = np.arange(np.sum(target['labels'] == label))
#         np.random.shuffle(sel)
#         target_train_idx.append(np.arange(len(target['labels']))[target['labels'] == label][sel[:ncovs_train[j]]])
#     target_train_idx = np.concatenate(target_train_idx)
#     target_test_idx  = np.array([i for i in range(len(target['labels'])) if i not in target_train_idx])

#     target_train = {}
#     target_train['covs'] = target['covs'][target_train_idx]
#     target_train['labels'] = target['labels'][target_train_idx]

#     target_test = {}
#     target_test['covs'] = target['covs'][target_test_idx]
#     target_test['labels'] = target['labels'][target_test_idx]

#     return source, target_train, target_test


def get_sourcetarget_split_motorimagery(target, ncovs_train, random_state):
    sss = StratifiedShuffleSplit(n_splits=1,
                                 train_size=2*ncovs_train/len(
                                     target['labels']
                                 ),
                                 random_state=random_state)
    for train_index, test_index in sss.split(target['covs'], target['labels']):
        target_train = {}
        target_train['covs'] = target['covs'][train_index]
        target_train['labels'] = target['labels'][train_index]

        target_test = {}
        target_test['covs'] = target['covs'][test_index]
        target_test['labels'] = target['labels'][test_index]

    return target_train, target_test


def get_sourcetarget_split_p300(source, target, ncovs_train):

    X_source = source['epochs']
    y_source = source['labels'].flatten()
    covs_source = ERPCovariances(
        classes=[2], estimator='lwf'
    ).fit_transform(X_source, y_source)

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
    idx_rest = np.where(y_target == 1)[0][:ncovs_train*5]
    # because there's one ERP in every 6 flashes

    idx_train = np.concatenate([idx_erps, idx_rest])
    idx_test = np.array(
        [i for i in range(len(y_target)) if i not in idx_train]
    )

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


# def transform_org2rct(source, target_train, target_test, weights_classes=None):

#     weights_source = np.ones(len(source['labels']))
#     weights_target = np.ones(len(target_train['labels']))
#     if weights_classes is not None:
#         for label in weights_classes.keys():
#             weights_source[source['labels'] == label] = weights_classes[label]
#             weights_target[
#                 target_train['labels'] == label
#             ] = weights_classes[label]

#     source_rct = {}
#     source_rct['labels'] = source['labels']
#     T = mean_riemann(source['covs'], sample_weight=weights_source)
#     T_source = np.stack([T]*len(source['covs']))
#     source_rct['covs'] = parallel_transport_covariances(
#         source['covs'], T_source
#     )

#     target_rct_train = {}
#     target_rct_train['labels'] = target_train['labels']
#     M_train = mean_riemann(
#         target_train['covs'], sample_weight=weights_target
#     )
#     T_target = np.stack([M_train]*len(target_train['covs']))
#     target_rct_train['covs'] = parallel_transport_covariances(
#         target_train['covs'], T_target
#     )

#     target_rct_test = {}
#     target_rct_test['labels'] = target_test['labels']
#     M_test = M_train
#     T_target = np.stack([M_test]*len(target_test['covs']))
#     target_rct_test['covs'] = parallel_transport_covariances(
#         target_test['covs'], T_target
#     )

#     return source_rct, target_rct_train, target_rct_test


def transform_org2rct(source, target_train, target_test, weights_classes=None):

    weights_source = np.ones(len(source['labels']))
    weights_target = np.ones(len(target_train['labels']))
    if weights_classes is not None:
        for label in weights_classes.keys():
            weights_source[source['labels'] == label] = weights_classes[label]
            weights_target[
                target_train['labels'] == label
            ] = weights_classes[label]

    domains = ['source_domain']*len(
        source['labels']
    ) + ['target_domain']*len(
        target_train['labels']
    )
    covs_all = np.concatenate((source['covs'], target_train['covs']))
    labels_all = np.concatenate((source['labels'], target_train['labels']))
    _, labels_enc = encode_domains(covs_all, labels_all, domains)

    source_rct = {}
    target_train_rct = {}
    target_test_rct = {}
    source_rct['labels'] = source['labels']
    target_train_rct['labels'] = target_train['labels']
    target_test_rct['labels'] = target_test['labels']

    rct = TLCenter(target_domain='target_domain')
    X_rct = rct.fit_transform(covs_all, labels_enc)
    source_rct['covs'] = X_rct[:len(source['labels'])]
    target_train_rct['covs'] = X_rct[len(source['labels']):]
    target_test_rct['covs'] = rct.transform(target_test['covs'])

    return source_rct, target_train_rct, target_test_rct


# def transform_rct2str(source, target_train, target_test):

#     covs_source = source['covs']
#     covs_target_train = target_train['covs']
#     covs_target_test = target_test['covs']

#     source_pow = {}
#     source_pow['covs'] = source['covs']
#     source_pow['labels'] = source['labels']

#     n = covs_source.shape[1]
#     disp_source = np.sum(
#         [distance_riemann(covi,
#                           np.eye(n)) ** 2 for covi in covs_source]
#     ) / len(covs_source)
#     disp_target = np.sum(
#         [distance_riemann(
#             covi, np.eye(n)
#          ) ** 2 for covi in covs_target_train]
#     ) / len(covs_target_train)
#     p = np.sqrt(disp_target / disp_source)

#     target_pow_train = {}
#     target_pow_train['covs'] = np.stack(
#         [powm(covi, 1.0/p) for covi in covs_target_train]
#     )
#     target_pow_train['labels'] = target_train['labels']

#     target_pow_test = {}
#     target_pow_test['covs'] = np.stack(
#         [powm(covi, 1.0/p) for covi in covs_target_test]
#     )
#     target_pow_test['labels'] = target_test['labels']

#     return source_pow, target_pow_train, target_pow_test


def transform_rct2str(source, target_train, target_test):

    domains = ['source_domain']*len(
        source['labels']
    ) + ['target_domain']*len(
        target_train['labels']
    )
    covs_all = np.concatenate((source['covs'], target_train['covs']))
    labels_all = np.concatenate((source['labels'], target_train['labels']))
    _, labels_enc = encode_domains(covs_all, labels_all, domains)

    source_str = {}
    target_train_str = {}
    target_test_str = {}
    source_str['labels'] = source['labels']
    target_train_str['labels'] = target_train['labels']
    target_test_str['labels'] = target_test['labels']

    str = TLStretch(target_domain='target_domain')
    X_str = str.fit_transform(covs_all, labels_enc)
    source_str['covs'] = X_str[:len(source['labels'])]
    target_train_str['covs'] = X_str[len(source['labels']):]
    target_test_str['covs'] = str.transform(target_test['covs'])

    return source_str, target_train_str, target_test_str


# def transform_str2rot(source, target_train, target_test,
#                       weights_classes=None, distance='euc'):

#     source_rot = {}
#     source_rot['covs'] = source['covs']
#     source_rot['labels'] = source['labels']

#     target_rot_train = {}
#     target_rot_train['labels'] = target_train['labels']

#     target_rot_test = {}
#     target_rot_test['labels'] = target_test['labels']

#     class_labels = np.unique(source['labels'])

#     M_source = []
#     for i in class_labels:
#         M_source_i = mean_riemann(source['covs'][source['labels'] == i])
#         M_source.append(M_source_i)

#     M_target_train = []
#     for j in class_labels:
#         M_target_train_j = mean_riemann(
#             target_train['covs'][target_train['labels'] == j]
#         )
#         M_target_train.append(M_target_train_j)

#     if weights_classes is None:
#         weights = [1] * len(class_labels)
#     else:
#         weights = []
#         for label in class_labels:
#             weights.append(weights_classes[label])

#     R = manifoptim.get_rotation_matrix(M=M_source, Mtilde=M_target_train,
#                                        dist=distance, weights=weights)

#     covs_target_train = np.stack(
#         [np.dot(R, np.dot(covi, R.T)) for covi in target_train['covs']]
#     )
#     target_rot_train['covs'] = covs_target_train

#     covs_target_test = np.stack(
#         [np.dot(R, np.dot(covi, R.T)) for covi in target_test['covs']]
#     )
#     target_rot_test['covs'] = covs_target_test

#     return source_rot, target_rot_train, target_rot_test


def transform_str2rot(source, target_train, target_test,
                      weights_classes=None, distance='euc'):

    domains = ['source_domain']*len(
        source['labels']
    ) + ['target_domain']*len(
        target_train['labels']
    )
    covs_all = np.concatenate((source['covs'], target_train['covs']))
    labels_all = np.concatenate((source['labels'], target_train['labels']))
    _, labels_enc = encode_domains(covs_all, labels_all, domains)

    source_rot = {}
    target_train_rot = {}
    target_test_rot = {}
    source_rot['labels'] = source['labels']
    target_train_rot['labels'] = target_train['labels']
    target_test_rot['labels'] = target_test['labels']

    rot = TLRotate(target_domain='target_domain')
    X_rot = rot.fit_transform(covs_all, labels_enc)
    source_rot['covs'] = X_rot[:len(source['labels'])]
    target_train_rot['covs'] = X_rot[len(source['labels']):]
    target_test_rot['covs'] = rot.transform(target_test['covs'])

    return source_rot, target_train_rot, target_test_rot
