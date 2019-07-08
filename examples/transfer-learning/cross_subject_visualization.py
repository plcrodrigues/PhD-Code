  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:28:33 2017

@author: coelhorp
"""

from riemann_lab import diffusion_maps as DM
from riemann_lab import transfer_learning as TL
from riemann_lab import get_datasets as GD

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.externals import joblib
from tqdm import tqdm

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann

# which dataset to consider
dataset = 'Cho2017'
settings = GD.get_settings(dataset, storage='GIPSA')
session = settings['session']
storage = settings['storage']

# load the source and target subjects dataset (it comes with signals and labels)
subject_target = 3
target_org = GD.get_dataset(dataset, subject_target, session, storage)
subject_source = 14
source_org = GD.get_dataset(dataset, subject_source, session, storage)

# estimates the Covariances for the source and target get_datasets
source_org['covs'] = Covariances(estimator='oas').fit_transform(source_org['signals'])
target_org['covs'] = Covariances(estimator='oas').fit_transform(target_org['signals'])

# number of trials in the target_train dataset (choosing maximal since we're not interested in classification)
ncovs = 99

# splitting the target dataset into training and testing for compatibility reasons with other code
source_org, target_org_train, target_org_test = TL.get_sourcetarget_split(source_org, target_org, ncovs)

# create a dictionary for stocking the embeddings
u = OrderedDict()
labs = np.concatenate([source_org['labels'], target_org['labels']])

# embedding of the original points
covs_org = np.concatenate([source_org['covs'], target_org['covs']])
u['org'],l = DM.get_diffusionEmbedding(points=covs_org, distance=distance_riemann)

# embedding of the re-centered points
source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source_org, target_org_train, target_org_test)
covs_rct = np.concatenate([source_rct['covs'], target_rct_train['covs'], target_rct_test['covs']])
u['rct'],l = DM.get_diffusionEmbedding(points=covs_rct, distance=distance_riemann)

# embedding of the stretched points
source_str, target_str_train, target_str_test = TL.RPA_stretch(source_rct, target_rct_train, target_rct_test)
covs_str = np.concatenate([source_str['covs'], target_str_train['covs'], target_str_test['covs']])
u['str'],l = DM.get_diffusionEmbedding(points=covs_str, distance=distance_riemann)

# embedding of the rotated points (output of RPA)
source_rot, target_rot_train, target_rot_test = TL.RPA_rotate(source_str, target_str_train, target_str_test)
covs_rot = np.concatenate([source_rot['covs'], target_rot_train['covs'], target_rot_test['covs']])
u['rot'],l = DM.get_diffusionEmbedding(points=covs_rot, distance=distance_riemann)

# scatter plot of the results
fig, ax = plt.subplots(facecolor='white', figsize=(11.46, 10.84), ncols=2, nrows=2)
plt.subplots_adjust(wspace=0.15, hspace=0.15)
ntrials = len(source_org['covs'])
sess = np.array(['source']*ntrials + ['target']*ntrials)
colors_sessions = {'source':'b', 'target':'r'}
for axi, methi in zip(ax.flatten(order='C'), u.keys()):
    for ui, si in zip(u[methi], sess):
        axi.scatter(ui[1], ui[2], c=colors_sessions[si], s=100, edgecolors='none')
        axi.set_title(methi, fontsize=20)
        axi.set_xticks([])
        axi.set_yticks([])
fig.show()
