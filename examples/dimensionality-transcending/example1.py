# %%
import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from moabb.datasets import Zhou2016, BNCI2015_001
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from utilities import dimensionality_transcending as DT
from utilities import transfer_learning as TL
# from sklearn.model_selection import KFold
# from tqdm import tqdm
from joblib import Parallel, delayed

plt.close('all')
# %%
# setup the paradigm
events = ["right_hand", "feet"]
paradigm = MotorImagery(events=events, n_classes=len(events))
paradigm_name = 'MI'

# set the weights for each class in the dataset
weights_classes = {}
weights_classes['feet'] = 1
weights_classes['right_hand'] = 1

# get data from source
source = {}
dataset_source = Zhou2016()
subject_source = 1
raw_source = dataset_source._get_single_subject_data(
    subject_source
)['session_0']['run_0']
# raw_source.pick_types(eeg=True)
raw_source.pick('eeg')
X, labels, meta = paradigm.get_data(dataset_source, subjects=[subject_source])
source['org'] = {}
# source['org']['covs'] = Covariances(estimator='lwf').fit_transform(X[:, 2:, :])
source['org']['covs'] = Covariances(estimator='lwf').fit_transform(X)
source['org']['labels'] = labels
source['org']['chnames'] = [chi.upper() for chi in raw_source.ch_names]
# source['org']['chnames'] = [chi.upper() for chi in raw_source.ch_names[2:]]

# get data from target
target = {}
dataset_target = BNCI2015_001()
subject_target = 2
raw_target = dataset_target._get_single_subject_data(
    subject_target
)['session_A']['run_0']
# raw_target.pick_types(eeg=True)
raw_target.pick('eeg')
X, labels, meta = paradigm.get_data(dataset_target, subjects=[subject_target])
target['org'] = {}
target['org']['covs'] = Covariances(estimator='lwf').fit_transform(X)
target['org']['labels'] = labels
target['org']['chnames'] = [chi.upper() for chi in raw_target.ch_names]

# get the indices of the electrode correspondances between the datasets
idx = DT.get_source_target_correspondance(source['org'], target['org'])

# setup the scores dictionary
# scores = {}
# # for meth in ['org-aug', 'rct-aug', 'str-aug', 'rot-aug', 'clb']:
# for meth in ['org-aug', 'rct-aug', 'str-aug', 'clb']:
#     scores[meth] = []

# %%
# split into training-testing dataset


def run_split(source, target, idx, ncovs_target_train, random_state):
    score = []
    target_train = {}
    target_test = {}

    (target_train['org'],
     target_test['org']) = TL.get_sourcetarget_split_motorimagery(
         target['org'], ncovs_target_train, random_state
    )
    # print(target_train['org']['labels'])
    # match the dimensionalities of the datasets
    (source['org-aug'],
     target_train['org-aug'],
     target_test['org-aug']) = DT.match_source_target_dimensions(
         source['org'],
         target_train['org'],
         target_test['org'],
         idx, paradigm_name='MI'
    )
    # apply RPA
    (source['rct-aug'],
     target_train['rct-aug'],
     target_test['rct-aug']) = TL.RPA_recenter(source['org-aug'],
                                               target_train['org-aug'],
                                               target_test['org-aug'],
                                               weights_classes)
    (source['str-aug'],
     target_train['str-aug'],
     target_test['str-aug']) = TL.RPA_stretch(source['rct-aug'],
                                              target_train['rct-aug'],
                                              target_test['rct-aug'])
    (source['rot-aug'],
     target_train['rot-aug'],
     target_test['rot-aug']) = TL.RPA_rotate(source['str-aug'],
                                             target_train['str-aug'],
                                             target_test['str-aug'],
                                             weights_classes)
    # which classifier to use
    clf = MDM()
    for meth in ['org-aug', 'rct-aug', 'str-aug', 'rot-aug']:
    # for meth in ['org-aug', 'rct-aug', 'str-aug']:
        auc = TL.get_score_transferlearning(clf,
                                            source[meth],
                                            target_train[meth],
                                            target_test[meth])
        score.append(dict(method=meth,
                          auc=auc,
                          seed=random_state))
    # get classification scores
    auc_clb = TL.get_score_calibration(clf, target_train['org'],
                                       target_test['org'])
    score.append(dict(method='clb',
                      auc=auc_clb,
                      seed=random_state))
    score = pd.DataFrame(score)
    return score


ncovs_target_train = 10

N_JOBS = 20
N_REPEATS = 400
rng = np.random.RandomState(42)
RANDOM_STATES = rng.randint(0, 10000, N_REPEATS)
scores = []
scores = Parallel(n_jobs=N_JOBS)(
    delayed(run_split)(source, target, idx, ncovs_target_train, random_state)
    for random_state in RANDOM_STATES
)

scores = pd.concat(scores)

# %% Plot results

sns.boxplot(data=scores, x='auc', y='method')
plt.title(
    f'Source: {dataset_source.__class__.__name__}, subject {subject_source} \n Target: {dataset_target.__class__.__name__}, subject {subject_target}')
plt.show()

# %% Tests
# ncovs_target_train = 10
# target_train = {}
# target_test = {}

# (target_train['org'],
#     target_test['org']) = TL.get_sourcetarget_split_motorimagery(
#         target['org'], ncovs_target_train, 42
# )
# # print(target_train['org']['labels'])
# # match the dimensionalities of the datasets
# (source['org-aug'],
#     target_train['org-aug'],
#     target_test['org-aug']) = DT.match_source_target_dimensions(
#         source['org'],
#         target_train['org'],
#         target_test['org'],
#         idx, paradigm_name='MI'
# )

# # %% plot the source and target covariance matrices
# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# _min, _max = source['org']['covs'][0].min(), source['org']['covs'][0].max()

# axes[0][0].imshow(source['org']['covs'][0], vmin=_min, vmax=_max)
# axes[0][0].set_title('Source covariance matrix')
# axes[0][1].imshow(source['org-aug']['covs'][0], vmin=_min, vmax=_max)
# axes[0][1].set_title(
#     'Source covariance matrix after\ndimensionality transcending'
# )
# axes[1][0].imshow(target_train['org']['covs'][0], vmin=_min, vmax=_max)
# axes[1][0].set_title('Target covariance matrix')
# im2 = axes[1][1].imshow(target_train['org-aug']['covs'][0],
#                         vmin=_min, vmax=_max)
# axes[1][1].set_title(
#     'Target covariance matrix after\ndimensionality transcending'
# )
# fig.colorbar(im2, ax=axes, shrink=0.6)
# plt.show()
# %%
