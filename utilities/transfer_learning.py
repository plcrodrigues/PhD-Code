#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:57:12 2018

@author: coelhorp
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from .helpers.transfer_learning.utils import transform_org2rct, transform_rct2str, transform_str2rot
from .helpers.transfer_learning.utils import get_sourcetarget_split_motorimagery, get_sourcetarget_split_p300

def get_sourcetarget_split(source, target, ncovs_train, paradigm='MI'):
    """Split the target dataset into a training and a testing set.

    This is a necessary step for assessing the classification performance
    in a semi-supervised paradigm with Transfer Learning. The handling of 
    P300 and MI epochs is slightly different, so we have an input argument 
    for selecting which one we want to consider.

    For MI data, the input arguments come with 'covs' already estimated.
    For P300 data, we need the 'ncovs_train' to know how many labeled epochs 
    are available for estimating the P300 prototype and the extended 
    covariances. The input arguments have 'epochs'.

    Parameters
    ----------
    source: dict, keys: MI is ['covs','labels'], P300 is ['epochs','labels']
    target: dict, keys: MI is ['covs','labels'], P300 is ['epochs','labels'] 
    ncovs_train : int
        how many labeled data points of each class to have in the training
        partition of the target dataset
    paradigm : string (default: 'MI')

    """        
    if paradigm == 'MI':
        return get_sourcetarget_split_motorimagery(source, target, ncovs_train)
    elif paradigm == 'P300':
        return get_sourcetarget_split_p300(source, target, ncovs_train)


def RPA_recenter(source, target_train, target_test, weights_classes=None):
    """Re-center the data points from source and target-train/target-test.

    This is the first step in the Riemannian Procrustes analysis (RPA)
    
    The re-centering is applied to both source and target data points, all of
    them are sent to the origin of the SPD manifold (i.e.,identity matrix).
    
    The re-centering matrix for the source dataset is calculated on all the 
    available dataset data points, whereas the re-centering for the target 
    dataset we use only the data points from the training partition.
    
    We may handle cases where the classes are unbalanced using the 
    weights_classes argument.

    Parameters
    ----------
    source: dict, keys: ['covs','labels'] 
    target_train: dict, keys: ['covs','labels']
    target_test: dict, keys: ['covs','labels']
    weights_classes : dict, keys: (names of classes)

    """    
    
    return transform_org2rct(source, target_train, target_test, weights_classes) 


def RPA_stretch(source, target_train, target_test):
    """Stretch the distribution of data points from target-train/target-test.

    This is the second step in the Riemannian Procrustes analysis (RPA)
    
    The stretching is applied to target data points so that their dispersion
    is the same as the one from the source data points
    
    The stretching factor is calculated using all the data points from the 
    source dataset and those from the target train partition.
    
    Parameters
    ----------
    source: dict, keys: ['covs','labels'] 
    target_train: dict, keys: ['covs','labels']
    target_test: dict, keys: ['covs','labels']

    """   

    return transform_rct2str(source, target_train, target_test)


def RPA_rotate(source, target_train, target_test, weights_classes=None, distance='euc'):
    """Rotate the distribution of data points from target-train/target-test.

    This is the third step in the Riemannian Procrustes analysis (RPA)
    
    The rotation is applied to target data points so that their class means
    are aligned with the class means from the source dataset
    
    The rotation matrix is calculated using only information from the data 
    points in the target train dataset
    
    Parameters
    ----------
    source: dict, keys: ['covs','labels'] 
    target_train: dict, keys: ['covs','labels']
    target_test: dict, keys: ['covs','labels']
    weights_classes : dict, keys: (names of classes)    

    """     
    
    return transform_str2rot(source, target_train, target_test, weights_classes, distance)

def get_score_calibration(clf, target_train, target_test):
    """Get the classification in calibration

    Training dataset: target_train
    Testing dataset: target_test
    
    Parameters
    ----------
    clf: classifier
    target_train: dict, keys: ['covs','labels']
    target_test: dict, keys: ['covs','labels']

    """  

    covs_train = target_train['covs']
    y_train = target_train['labels']
    covs_test = target_test['covs']
    y_test = target_test['labels']

    clf.fit(covs_train, y_train)

    y_pred = clf.predict(covs_test)

    y_test = np.array([y_test == i for i in np.unique(y_test)]).T
    y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T

    return roc_auc_score(y_test, y_pred)

def get_score_transferlearning(clf, source, target_train, target_test):
    """Get the transfer learning score
    
    Training dataset: target_train + source
    Testing dataset: target_test

    Parameters
    ----------
    clf: classifier
    source: dict, keys: ['covs','labels']    
    target_train: dict, keys: ['covs','labels']
    target_test: dict, keys: ['covs','labels']

    """     

    covs_source, y_source = source['covs'], source['labels']
    covs_target_train, y_target_train = target_train['covs'], target_train['labels']
    covs_target_test, y_target_test = target_test['covs'], target_test['labels']

    covs_train = np.concatenate([covs_source, covs_target_train])
    y_train = np.concatenate([y_source, y_target_train])
    clf.fit(covs_train, y_train)

    covs_test = covs_target_test
    y_test = y_target_test

    y_pred = clf.predict(covs_test)

    y_test = np.array([y_test == i for i in np.unique(y_test)]).T
    y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T

    return roc_auc_score(y_test, y_pred)

