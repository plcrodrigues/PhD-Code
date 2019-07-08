#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:36:09 2018

@author: coelhorp
"""

import numpy as np
import pandas as pd

from collections import OrderedDict
from itertools import combinations

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from pyriemann.classification import MDM
from riemann_lab.dimensionality_reduction import RDR
from statsmodels.stats.multitest import multipletests

from riemann_lab import get_datasets as GD
from riemann_lab import statistics as ST

def score_cross_validation(clf, covs, labs, n_splits=10):

    scores = []

    kf = StratifiedKFold(n_splits=n_splits)

    for train_idx, test_idx in kf.split(covs, labs):
        covs_train, labs_train = covs[train_idx], labs[train_idx]
        covs_test, labs_test = covs[test_idx], labs[test_idx]
        clf.fit(covs_train, labs_train)
        scores.append(clf.score(covs_test, labs_test))

    return np.mean(scores), np.std(scores)

def get_scores_loop(storage):

    # which dataset to consider
    dataset = 'PhysionetMI'
    nsubj = 109

    scores = {}
    for method in ['full', 'select', 'covpca', 'gpcaRiemann', 'gpcaEuclid']:
        scores[method] = []

    # load the subject dataset
    for subject in range(1, nsubj+1):

        print('subject', subject)

        # dimension to which reduce the SPD matrices
        p = 12        

        # original dataset with 64 electrodes
        data = GD.get_dataset_physionet_covs(subject, full=True, storage=storage)
        covs_full = data['covs']; labs = data['labels']
        pipeline = make_pipeline(MDM())
        score_full = score_cross_validation(pipeline, covs_full, labs)
        scores['full'].append(score_full)
        print('full -', 'mean:', score_full[0], 'std:', score_full[1])

        # selected 12 electrodes with physiological meaning
        data = GD.get_dataset_physionet_covs(subject, full=False, storage=storage)
        covs_select = data['covs']; labs = data['labels']
        pipeline = make_pipeline(MDM())
        score_full = score_cross_validation(pipeline, covs_select, labs)
        scores['select'].append(score_full)
        print('select -', 'mean:', score_full[0], 'std:', score_full[1])

        # dimensionality reduction via cov-PCA
        rdr = RDR(n_components=p, method='covpca')
        pipeline = make_pipeline(rdr, MDM())
        score_covpca = score_cross_validation(pipeline, covs_full, labs)
        scores['covpca'].append(score_covpca)
        print('covpca -', 'mean:', score_covpca[0], 'std:', score_covpca[1])

        # dimensionality reduction via generalized PCA with Riemannian distance
        rdr = RDR(n_components=p, method='gpcaRiemann')
        pipeline = make_pipeline(rdr, MDM())
        score_gpca_riemann = score_cross_validation(pipeline, covs_full, labs)
        scores['gpcaRiemann'].append(score_gpca_riemann)
        print('gpcaRiemann -', 'mean:', score_gpca_riemann[0], 'std:', score_gpca_riemann[1])

        # dimensionality reduction via generalized PCA with Euclidean distance
        rdr = RDR(n_components=p, method='gpcaEuclid')
        pipeline = make_pipeline(rdr, MDM())
        score_gpca_euclid = score_cross_validation(pipeline, covs_full, labs)
        scores['gpcaEuclid'].append(score_gpca_euclid)
        print('gpcaEuclid -', 'mean:', score_gpca_euclid[0], 'std:', score_gpca_euclid[1])

        print('')

    filename = 'linear_methods_classification_physionet_scores.pkl'
    joblib.dump(scores, filename)

def get_stats_loop():

    filename = './results/linear_methods_classification_physionet_scores.pkl'
    scores = joblib.load(filename)

    df = pd.DataFrame()
    for meth in scores.keys():
        df[meth] = np.array([scores[meth][i][0] for i in range(len(scores[meth]))])

    nrzt = 10000
    stats = ST.permutations_paired_t_test(df, nrzt, multiple_correction=True)

    return stats

dataset = 'PhysionetMI'
storage = 'GIPSA'
#get_scores_loop(storage=storage)
stats = get_stats_loop()
print(dataset)
print(stats)
