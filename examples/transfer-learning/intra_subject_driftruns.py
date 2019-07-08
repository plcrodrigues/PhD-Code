#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:33:44 2018

@author: coelhorp
"""

from riemann_lab import diffusion_maps as DM
from riemann_lab import transfer_learning as TL
from riemann_lab import get_datasets as GD
from riemann_lab import classification as CL

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from sklearn.externals import joblib

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann

def plot_driftruns_gigadb(subject=1, storage='GIPSA'):

    # load the subject's dataset
    data = GD.get_dataset_gigadb_covs(subject, full=True, storage=storage)
    covs = data['covs']
    labs = data['labels']

    # embedding of the points
    u,l = DM.get_diffusionEmbedding(points=covs, distance=distance_riemann)

    # setup the figure
    fig = plt.figure(facecolor='white', figsize=(19.29, 9.08))
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 1))

    # scatter plots with results
    ax1.scatter(u[:,1], u[:,2], s=180, c=range(len(u)), cmap='viridis')
    ax2.plot(u[:,1], lw=2.0, c='k')
    ax3.plot(u[:,2], lw=2.0, c='k')
    ax4.plot(u[:,3], lw=2.0, c='k')

    fig.show()

def plot_driftruns_correction_gigadb(subject=1, storage='GIPSA'):

    # load the subject's dataset
    data_org = GD.get_dataset_gigadb_covs(subject, full=True, storage=storage)
    covs_org = data_org['covs']
    labs_org = data_org['labels']

    # embedding of the points
    u_org,l = DM.get_diffusionEmbedding(points=covs_org, distance=distance_riemann)

    # define a dict for the runs indices
    runs_dict = {}
    for runi in range(1, 5+1):
        runs_dict[runi] = (runi-1)*40 + np.arange(40)

    # define a dict for the runs indices on the GigaDB dataset
    data_rct = TL.RPA_correct_driftruns(data_org, runs_dict)
    covs_rct = data_rct['covs']
    labs_rct = data_rct['labels']

    # embedding of the points
    u_rct,l = DM.get_diffusionEmbedding(points=covs_rct, distance=distance_riemann)

    # setup the figure
    fig = plt.figure(facecolor='white', figsize=(19.29, 9.08))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    # scatter plots with results
    ax1.scatter(u_org[:,1], u_org[:,2], s=180, c=range(len(u_org)), cmap='viridis')
    ax2.scatter(u_rct[:,1], u_rct[:,2], s=180, c=range(len(u_rct)), cmap='viridis')

    fig.show()

def get_scores_driftruns_correction_gigadb(storage='GIPSA'):

    scores = {}

    # load the subject dataset
    for file in glob('/research/vibs/Pedro/datasets/moabb/Cho2017/covsfull/*'):

        subject = int(file.split('/')[-1].strip('subject_').strip('_full.pkl'))
        scores[subject] = {}

        # load the subject's dataset
        data_org = GD.get_dataset_gigadb_covs(subject, full=True, storage=storage)
        covs_org = data_org['covs']
        labs_org = data_org['labels']

        # define a dict for the runs indices on the GigaDB dataset
        runs_dict = {}
        for runi in range(1, 5+1):
            runs_dict[runi] = (runi-1)*40 + np.arange(40)

        # correct for the drifts on each run
        data_rct = TL.RPA_correct_driftruns(data_org, runs_dict)
        covs_rct = data_rct['covs']
        labs_rct = data_rct['labels']

        scores[subject]['org'] = CL.score_cross_validation(MDM(), data_org)
        scores[subject]['rct'] = CL.score_cross_validation(MDM(), data_rct)

        print(subject)
        print('org:', scores[subject]['org'])
        print('rct:', scores[subject]['rct'])
        print('')

    filename = './intra_subject_driftruns_scores.pkl'
    joblib.dump(scores, filename)

    return scores

storage = 'GIPSA'
#plot_driftruns_gigadb(subject=1, storage=storage)
#plot_driftruns_correction_gigadb(subject=1, storage=storage)
scores = get_scores_driftruns_correction_gigadb(storage=storage)
