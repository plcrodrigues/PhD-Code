import sys
sys.path.append('../../')

import numpy as np
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from utilities import dimensionality_reduction as DR
from sklearn.model_selection import KFold
from tqdm import tqdm

# setup which dataset to consider from MOABB
dataset = PhysionetMI()
paradigm = MotorImagery()
paradigm_name = 'MI'

# choose dimension to reduce
pred = 12

# setup the the classifier
clf = MDM()

# which subject to consider
subject = 7

# load data	
X, labels, meta = paradigm.get_data(dataset, subjects=[subject])
covs = Covariances(estimator='lwf').fit_transform(X)

# get the indices for the electrodes chosen in SELg and SELb
raw = dataset._get_single_subject_data(subject)['session_0']['run_4']
chnames_dict = {}
for i, chi in enumerate(raw.ch_names):
	chnames_dict[chi.upper()] = i
SELg_names = ['F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'P4']
SELg = [chnames_dict[chi] for chi in SELg_names]
SELb_names = ['FPZ', 'FP1', 'AFZ', 'AF3', 'AF7', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3']
SELb = [chnames_dict[chi] for chi in SELb_names]

# setup the scores dictionary
scores = {}
for meth in ['covpca', 'gpcaRiemann', 'SELg', 'SELb']:
	scores[meth] = []

# do a KFold loop to get the scores
n_splits = 5
kf = KFold(n_splits=n_splits)
for train_index, test_index in tqdm(kf.split(covs), total=n_splits):

	# split into training and testing datasets
	covs_train = covs[train_index]
	labs_train = labels[train_index]
	covs_test = covs[test_index]
	labs_test = labels[test_index]

	# reduce the dimensions with ['covpca', 'gpcaRiemann']
	for meth in ['covpca', 'gpcaRiemann']:
		trf = DR.RDR(n_components=pred, method=meth)
		trf.fit(covs_train)
		covs_train_red = trf.transform(covs_train)
		covs_test_red = trf.transform(covs_test)
		clf.fit(covs_train_red, labs_train)
		scores[meth].append(clf.score(covs_test_red, labs_test))

	# reduce the dimensions with [SELg, SELb]
	for meth, sel in zip(['SELg', 'SELb'], [SELg, SELb]):
		covs_train_red = covs_train[:, sel, :][:, :, sel]
		covs_test_red = covs_test[:, sel, :][:, :, sel]
		clf.fit(covs_train_red, labs_train)
		scores[meth].append(clf.score(covs_test_red, labs_test))

print('subject ', subject)
# print the scores
for meth in scores.keys():
	print(meth, np.mean(scores[meth]))
print('')	


