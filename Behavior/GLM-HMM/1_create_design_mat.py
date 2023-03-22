#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:42:14 2022
By: Guido Meijer
"""

import numpy as np
from sklearn import preprocessing
import numpy.random as npr
import os
import json
from collections import defaultdict
from os.path import join, isdir
from sklearn import preprocessing
from data_utils import paths, search_sessions_by_tag, search_via_insertions, behavioral_criterion
from glm_hmm_utils import get_all_unnormalized_data_this_session, fit_glm, create_train_test_sessions
from one.api import ONE
one = ONE()

from pprint import pprint

"""
Adapted from on Zoe Ashwood's code (https://github.com/zashwood/glm-hmm)
"""

# Settings
N_FOLDS = 5
MIN_SESSIONS = 2

npr.seed(42)

# Paths
_, save_path = paths()
save_path = join(save_path, 'GLM-HMM')

# Create folders
os.makedirs(save_path, exist_ok=True)
os.makedirs(join(save_path, 'data_by_animal'), exist_ok=True)


'''
    1. search all sessions in the with qc ephys (E) and behaviour (B)
       and build a dictionary of sessions by subject
'''
dict_path = join(save_path, "EB_animal_eid_dict.json")
try:
    with open(dict_path, "r") as f:
        EB_animal_eid_dict = json.load(f)
except Exception as err:
    print(err)
    EB_animal_eid_dict = search_via_insertions(one=one)
    with open(dict_path, "w") as f:
        f.write(json.dumps(EB_animal_eid_dict))
temp_animal_eid_dict, _ = EB_animal_eid_dict


'''
    2. Build a dictionary with all the (good quality) behavioural
       sessions for the animals found above
'''
animal_eid_dict = dict()
subjects = list(temp_animal_eid_dict.keys())
print(len(subjects))
for i, subject in enumerate(subjects):

    print(subject)
    # query session per animal
    sessions = one.alyx.rest('sessions', 'list',
                django='session__task_protocol__icontains,choiceworld,'
                       f'subject__nickname__in,{[subject]}')
    eids = [sess['id'] for sess in sessions]
    if len(eids) > 0:
        print(f"\t{len(eids)} sessions")
        eids = behavioral_criterion(eids, max_lapse=0.4, max_bias=0.5, min_trials=200, one=one)

        # animal_eid_dict is a dict with subjects as keys and a list of eids per subject
        animal_eid_dict[subject] = eids
    else:
        print("\tno sessions")

# Require that each animal has enough sessions
animal_list = []
for subject in list(animal_eid_dict.keys()):
    num_sessions = len(animal_eid_dict[subject])
    if num_sessions < MIN_SESSIONS:
        continue
    animal_list.append(subject)


# %%
# Identify idx in master array where each animal's data starts and ends:
animal_start_idx = {}
animal_end_idx = {}

final_animal_eid_dict = defaultdict(list)
# WORKHORSE: iterate through each animal and each animal's set of eids;
# obtain unnormalized data.  Write out each animal's data and then also
# write to master array
for z, animal in enumerate(animal_list):
    print(animal)
    sess_counter = 0
    for eid in animal_eid_dict[animal]:
        print("\t",eid)
        try:
            animal, unnormalized_inpt, y, session, num_viols_50, rewarded = \
                get_all_unnormalized_data_this_session(
                    eid, one)
            if num_viols_50 < 10:  # only include session if number of viols is less than 10
                if sess_counter == 0:
                    animal_unnormalized_inpt = np.copy(unnormalized_inpt)
                    animal_y = np.copy(y)
                    animal_session = session
                    animal_rewarded = np.copy(rewarded)
                else:
                    animal_unnormalized_inpt = np.vstack(
                        (animal_unnormalized_inpt, unnormalized_inpt))
                    animal_y = np.vstack((animal_y, y))
                    animal_session = np.concatenate((animal_session, session))
                    animal_rewarded = np.vstack((animal_rewarded, rewarded))
                sess_counter += 1
                final_animal_eid_dict[animal].append(eid)
        except Exception as err:
            print(err)
    # Write out animal's unnormalized data matrix:
    np.savez(join(save_path, 'data_by_animal', animal + '_unnormalized.npz'),
             animal_unnormalized_inpt, animal_y, animal_session)
    animal_session_fold_lookup = create_train_test_sessions(animal_session, N_FOLDS)
    np.savez(join(save_path, 'data_by_animal', animal + "_session_fold_lookup.npz"),
             animal_session_fold_lookup)
    np.savez(join(save_path, 'data_by_animal', animal + '_rewarded.npz'),
             animal_rewarded)
    assert animal_rewarded.shape[0] == animal_y.shape[0]
    # Now create or append data to master array across all animals:
    if z == 0:
        master_inpt = np.copy(animal_unnormalized_inpt)
        animal_start_idx[animal] = 0
        animal_end_idx[animal] = master_inpt.shape[0] - 1
        master_y = np.copy(animal_y)
        master_session = animal_session
        master_session_fold_lookup_table = animal_session_fold_lookup
        master_rewarded = np.copy(animal_rewarded)
    else:
        animal_start_idx[animal] = master_inpt.shape[0]
        master_inpt = np.vstack((master_inpt, animal_unnormalized_inpt))
        animal_end_idx[animal] = master_inpt.shape[0] - 1
        master_y = np.vstack((master_y, animal_y))
        master_session = np.concatenate((master_session, animal_session))
        master_session_fold_lookup_table = np.vstack(
            (master_session_fold_lookup_table, animal_session_fold_lookup))
        master_rewarded = np.vstack((master_rewarded, animal_rewarded))
# Write out data from across animals
assert np.shape(master_inpt)[0] == np.shape(master_y)[
    0], "inpt and y not same length"
assert np.shape(master_rewarded)[0] == np.shape(master_y)[
    0], "rewarded and y not same length"
assert len(np.unique(master_session)) == \
       np.shape(master_session_fold_lookup_table)[
           0], "number of unique sessions and session fold lookup don't " \
               "match"
normalized_inpt = np.copy(master_inpt)
normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0])
np.savez(join(save_path, 'all_animals_concat.npz'),
         normalized_inpt, master_y, master_session)
np.savez(join(save_path, 'all_animals_concat_unnormalized.npz'),
              master_inpt, master_y, master_session)
np.savez(join(save_path, 'all_animals_concat_session_fold_lookup.npz'),
         master_session_fold_lookup_table)
np.savez(join(save_path, 'all_animals_concat_rewarded.npz'),
         master_rewarded)
np.savez(join(save_path, 'data_by_animal', 'animal_list.npz'),
         animal_list)

with open(join(save_path, "final_animal_eid_dict.json"), "w") as f:
    f.write(json.dumps(final_animal_eid_dict))

# Now write out normalized data (when normalized across all animals) for
# each animal:
counter = 0
for animal in animal_start_idx.keys():
    start_idx = animal_start_idx[animal]
    end_idx = animal_end_idx[animal]
    inpt = normalized_inpt[range(start_idx, end_idx + 1)]
    y = master_y[range(start_idx, end_idx + 1)]
    session = master_session[range(start_idx, end_idx + 1)]
    counter += inpt.shape[0]
    np.savez(join(save_path, 'data_by_animal', animal + '_processed.npz'),
             inpt, y, session)

assert counter == master_inpt.shape[0]