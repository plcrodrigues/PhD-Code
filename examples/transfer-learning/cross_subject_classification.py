  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:28:33 2017

@author: coelhorp
"""

from riemann_lab import transfer_learning as TL
from riemann_lab import get_datasets as GD

import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from sklearn.externals import joblib
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

def get_scores(clf, source, target, ncovs):

    # estimates the Covariances for the source and target get_datasets
    source['covs'] = Covariances(estimator='oas').fit_transform(source['signals'])
    target['covs'] = Covariances(estimator='oas').fit_transform(target['signals'])

    # create a scores dictionary
    methods_list = ['org', 'rct', 'str', 'rot']
    scores = OrderedDict()
    for method in methods_list:
        scores[method] = []

    nrzt = 10
    for _ in range(nrzt):

        # get the split for the source and target dataset
        source_org, target_org_train, target_org_test = TL.get_sourcetarget_split(source, target, ncovs)

        # get the score with the original dataset
        scores['org'].append(TL.get_score_transferlearning(clf, source_org, target_org_train, target_org_test))

        # get the score with the re-centered matrices
        source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source_org, target_org_train, target_org_test)
        scores['rct'].append(TL.get_score_transferlearning(clf, source_rct, target_rct_train, target_rct_test))

        # stretch the classes
        source_str, target_str_train, target_str_test = TL.RPA_stretch(source_rct, target_rct_train, target_rct_test)
        scores['str'].append(TL.get_score_transferlearning(clf, source_str, target_str_train, target_str_test))

        # rotate the re-centered-stretched matrices using information from classes
        source_rot, target_rot_train, target_rot_test = TL.RPA_rotate(source_str, target_str_train, target_str_test)
        scores['rot'].append(TL.get_score_transferlearning(clf, source_rot, target_rot_train, target_rot_test))

    for method in methods_list:
        scores[method] = np.mean(scores[method])

    return scores

# which dataset to consider
dataset = 'BNCI2014001'
settings = GD.get_settings(dataset, storage='GIPSA')
session = settings['session']
storage = settings['storage']

# which classifier to use
clf = MDM()

# load the source and target subjects dataset (it comes with signals and labels)
subject_target = 1
target = GD.get_dataset(dataset, subject_target, session, storage)
subject_source = 2
source = GD.get_dataset(dataset, subject_source, session, storage)

# how many labeled trials in the target dataset
ncovs = 12

# get the scores on different methods for Transfer Learning
scores = get_scores(clf, source, target, ncovs)

# print the scores
for key, value in scores.items():
    print(key, value)
