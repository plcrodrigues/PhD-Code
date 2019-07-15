import sys
sys.path.append('../../')

import numpy as np
from moabb.datasets import Cho2017
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from utilities import transfer_learning as TL
from tqdm import tqdm

# setup which dataset to consider from MOABB
dataset = Cho2017()
paradigm = MotorImagery()
paradigm_name = 'MI'

# which subjects to consider
subject_source = 1
subject_target = 3
ncovs_target_train = 20

# set the weights for each class in the dataset
weights_classes = {}
weights_classes['left_hand'] = 1
weights_classes['right_hand'] = 1

# get the data for the source and target subjects
data_source = {}
data_target = {}
X, labels, meta = paradigm.get_data(dataset, subjects=[subject_source])
data_source['covs'] = Covariances(estimator='lwf').fit_transform(X)
data_source['labels'] = labels
X, labels, meta = paradigm.get_data(dataset, subjects=[subject_target])
data_target['covs'] = Covariances(estimator='lwf').fit_transform(X)
data_target['labels'] = labels

# setup the scores dictionary
scores = {}
for meth in ['org', 'rct', 'str', 'rot', 'clb']:
	scores[meth] = []

# apply RPA to multiple random partitions for the training dataset
clf = MDM()
nrzt = 5
for _ in tqdm(range(nrzt)):

	# split the target dataset into training and testing
	source = {}
	target_train = {}
	target_test = {}
	source['org'], target_train['org'], target_test['org'] = TL.get_sourcetarget_split(data_source, data_target, ncovs_target_train, paradigm=paradigm_name)

	# apply RPA 
	source['rct'], target_train['rct'], target_test['rct'] = TL.RPA_recenter(source['org'], target_train['org'], target_test['org'], weights_classes)
	source['str'], target_train['str'], target_test['str'] = TL.RPA_stretch(source['rct'], target_train['rct'], target_test['rct'])
	source['rot'], target_train['rot'], target_test['rot'] = TL.RPA_rotate(source['str'], target_train['str'], target_test['str'], weights_classes)

	# get classification scores
	scores['clb'].append(TL.get_score_calibration(clf, target_train['org'], target_test['org']))
	for meth in source.keys():
		scores[meth].append(TL.get_score_transferlearning(clf, source[meth], target_train[meth], target_test[meth]))

# print the scores
for meth in scores.keys():
	print(meth, np.mean(scores[meth]))	
	
