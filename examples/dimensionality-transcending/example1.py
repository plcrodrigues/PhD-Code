import sys
sys.path.append('../../')

import numpy as np
from moabb.datasets import Zhou2016, BNCI2015001
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from utilities import dimensionality_transcending as DT
from utilities import transfer_learning as TL
from sklearn.model_selection import KFold
from tqdm import tqdm

# setup the paradigm
paradigm = MotorImagery(events=['right_hand', 'feet'])
paradigm_name = 'MI'

# set the weights for each class in the dataset
weights_classes = {}
weights_classes['feet'] = 1
weights_classes['right_hand'] = 1

# get data from source 
source = {}
dataset_source = Zhou2016()
subject_source = 1
raw_source = dataset_source._get_single_subject_data(subject_source)['session_0']['run_0']
raw_source.pick_types(eeg=True)
X, labels, meta = paradigm.get_data(dataset_source, subjects=[subject_source])
source['org'] = {}
source['org']['covs'] = Covariances(estimator='lwf').fit_transform(X[:,2:,:])
source['org']['labels'] = labels
source['org']['chnames'] = [chi.upper() for chi in raw_source.ch_names[2:]]

# get data from target 
target = {}
dataset_target = BNCI2015001()
subject_target = 1
raw_target = dataset_target._get_single_subject_data(subject_target)['session_A']['run_0']
raw_target.pick_types(eeg=True)
X, labels, meta = paradigm.get_data(dataset_target, subjects=[subject_target])
target['org'] = {}
target['org']['covs'] = Covariances(estimator='lwf').fit_transform(X)
target['org']['labels'] = labels
target['org']['chnames'] = [chi.upper() for chi in raw_target.ch_names]

# get the indices of the electrode correspondances between the datasets
idx = DT.get_source_target_correspondance(source['org'], target['org'])

# setup the scores dictionary
scores = {}
for meth in ['org-aug', 'rct-aug', 'str-aug', 'rot-aug', 'clb']:
	scores[meth] = []

# which classifier to use
clf = MDM()

# split into training-testing dataset
ncovs_target_train = 10
nrzt = 5
for _ in tqdm(range(nrzt)):

	target_train = {}
	target_test = {}
	source['org'], target_train['org'], target_test['org'] = TL.get_sourcetarget_split(source['org'], target['org'], ncovs_target_train, paradigm=paradigm_name)

	# match the dimensionalities of the datasets
	source['org-aug'], target_train['org-aug'], target_test['org-aug'] = DT.match_source_target_dimensions(source['org'], target_train['org'], target_test['org'], idx, paradigm_name='MI')

	# apply RPA 
	source['rct-aug'], target_train['rct-aug'], target_test['rct-aug'] = TL.RPA_recenter(source['org-aug'], target_train['org-aug'], target_test['org-aug'], weights_classes)
	source['str-aug'], target_train['str-aug'], target_test['str-aug'] = TL.RPA_stretch(source['rct-aug'], target_train['rct-aug'], target_test['rct-aug'])
	source['rot-aug'], target_train['rot-aug'], target_test['rot-aug'] = TL.RPA_rotate(source['str-aug'], target_train['str-aug'], target_test['str-aug'], weights_classes)

	# get classification scores
	scores['clb'].append(TL.get_score_calibration(clf, target_train['org'], target_test['org']))
	for meth in ['org-aug', 'rct-aug', 'str-aug', 'rot-aug']:
		scores[meth].append(TL.get_score_transferlearning(clf, source[meth], target_train[meth], target_test[meth]))

for meth in scores.keys():
	print(meth, np.mean(scores[meth]))



