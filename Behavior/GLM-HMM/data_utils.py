import json
import numpy as np
import pandas as pd
from os.path import join, realpath, dirname, isfile
from one.api import ONE
from pprint import pprint

def paths(dropbox=False):
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input
    """
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        paths = dict()
        paths['fig_path'] = input('Path folder to save figures: ')
        paths['save_path'] = input('Path folder to save data: ')
        paths['dropbox_path'] = input('Path to Dropbox folder (can be empty):')
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(paths, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        paths = json.load(json_file)
    if dropbox:
        fig_path = paths['dropbox_path']
    else:
        fig_path = paths['fig_path']
    save_path = paths['save_path']
    #save_path = join(dirname(realpath(__file__)), 'Data')
    return fig_path, save_path



def load_trials(eid, invert_choice=False, invert_stimside=False, one=None):
    if one is None:
        one = ONE()

    data = one.load_object(eid, 'trials')
    data = {your_key: data[your_key] for your_key in [
        'stimOn_times', 'feedback_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice', 'firstMovement_times']}
    trials = pd.DataFrame(data=data)
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    
    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
    trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
               'stim_side'] = 1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
               'stim_side'] = -1
    if 'firstMovement_times' in trials.columns.values:
        trials['reaction_times'] = trials['firstMovement_times'] - trials['goCue_times']
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']

    return trials


def search_via_insertions(one=None):
    if one is None:
        one = ONE()
    # 1. Load all probe insertions
    # pids = query_sessions_pids_all(selection='resolved-behavior')
    ins = one.alyx.rest('insertions', 'list',
            django= 'session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                    'session__qc__lt,50,'
                    'json__extended_qc__alignment_resolved,True,'
                    'session__extended_qc__behavior,1')


    # 2. Group sessions by subject
    subject_eid_dict = dict()
    for i, p in enumerate(ins):
        subject = p['session_info']['subject']
        eid = p['session']
        try:
            if eid not in subject_eid_dict[subject]:
                subject_eid_dict[subject].append(eid)
        except:
            subject_eid_dict[subject] = [eid]

    # 3. Group probe insertions by session
    eid_pid_dict = dict()
    for i, p in enumerate(ins):
        eid = p['session']
        pid = p['id']
        try:
            if eid not in eid_pid_dict[eid]:
                eid_pid_dict[eid].append(pid)
        except:
            eid_pid_dict[eid] = [pid]

    return subject_eid_dict, eid_pid_dict

def search_sessions_by_tag (tag='2022_Q4_IBL_et_al_BWM', one=None):
    if one is None:
        one = ONE()

    sessions = one.alyx.rest('sessions', 'list', tag=tag)
    # pprint(sessions[0])

    subject_eid_dict = dict()
    for i, sess in enumerate(sessions):
        subject = sess['subject']
        eid = sess['id']
        try:
            if eid not in subject_eid_dict[subject]:
                subject_eid_dict[subject].append(eid)
        except KeyError:
            subject_eid_dict[subject] = [eid]
    return subject_eid_dict


def behavioral_criterion(eids, max_lapse=0.5, max_bias=0.5, min_trials=200, return_excluded=False,
                         one=None):
    if one is None:
        one = ONE()
    use_eids, excl_eids = [], []
    for j, eid in enumerate(eids):
        try:
            trials = load_trials(eid, one=one)
            lapse_l = 1 - (np.sum(trials.loc[trials['signed_contrast'] == -1, 'choice'] == 1)
                           / trials.loc[trials['signed_contrast'] == -1, 'choice'].shape[0])
            lapse_r = 1 - (np.sum(trials.loc[trials['signed_contrast'] == 1, 'choice'] == -1)
                           / trials.loc[trials['signed_contrast'] == 1, 'choice'].shape[0])
            bias = np.abs(0.5 - (np.sum(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)
                                 / np.shape(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)[0]))
            details = one.get_details(eid)
            if ((lapse_l < max_lapse) & (lapse_r < max_lapse) & (trials.shape[0] > min_trials)
                    & (bias < max_bias)):
                use_eids.append(eid)
            else:
                print('%s %s excluded (n_trials: %d, lapse_l: %.2f, lapse_r: %.2f, bias: %.2f)'
                      % (details['subject'], details['start_time'][:10], trials.shape[0], lapse_l, lapse_r, bias))
                excl_eids.append(eid)
        except Exception as e:
            print(e)
            print('Could not load session %s' % eid)
    if return_excluded:
        return use_eids, excl_eids
    else:
        return use_eids
