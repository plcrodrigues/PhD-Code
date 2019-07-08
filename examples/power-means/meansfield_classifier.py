#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:32:42 2018

@author: coelhorp
"""

import numpy as np
from tqdm import tqdm

from pyriemann.classification import MDM
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.estimation import Covariances

from riemann_lab import get_datasets as GD
from riemann_lab import power_means

# choose which dataset and subject to consider
dataset = 'BNCI2014001'
storage = 'GIPSA'
settings = GD.get_settings(dataset, storage)
session = settings['session']

# load the data
subject = 1
data = GD.get_dataset(dataset, subject, session, storage)
covs = Covariances(estimator='oas').fit_transform(data['signals'])
labs = data['labels']

# define the pipelines for classification -- MDM and MeansField classifier
pipelines = {}
pipelines['MDM'] = MDM()
plist = [1.00, 0.75, 0.50, 0.25, 0.10, 0.01, -0.01, -0.10, -0.25, -0.50, -0.75, -1.00]
pipelines['MeansField'] = power_means.MeanFieldClassifier(plist=plist)

# perform the KFold cross-validation procedure with stratified segments
# (same proportion of labels form each class on every fold)
n_splits = 5
kf = StratifiedKFold(n_splits)
scores = {}
for pipeline_name in pipelines.keys():
    scores[pipeline_name] = 0
for train_idx, test_idx in tqdm(kf.split(covs, labs), total=n_splits):
    covs_train, labs_train = covs[train_idx], labs[train_idx]
    covs_test, labs_test = covs[test_idx], labs[test_idx]
    for pipeline_name in pipelines.keys():
        pipelines[pipeline_name].fit(covs_train, labs_train)
        y_pred = pipelines[pipeline_name].predict(covs_test)
        y_test = np.array([labs_test == i for i in np.unique(labs_test)]).T
        y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T
        scores[pipeline_name] += roc_auc_score(y_test, y_pred)

for pipeline_name in pipelines.keys():
    scores[pipeline_name] = scores[pipeline_name] / (n_splits)

print('')
print(scores)
