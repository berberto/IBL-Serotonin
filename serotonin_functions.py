# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import tkinter as tk
import patsy
import statsmodels.api as sm
from sklearn.model_selection import KFold
from os.path import join
from glob import glob
from datetime import datetime
from brainbox.io.spikeglx import spikeglx
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
from one.api import ONE

# This is the date at which I fixed a crucial bug in the laser driver, all optogenetics data
# from before this date is not to be trusted
DATE_GOOD_OPTO = '2021-07-01'

# This is the date at which light shielding was added so that the mouse couldn't see the opto stim
DATE_LIGHT_SHIELD = '2021-06-08'


def load_subjects(behavior=None):
    subjects = pd.read_csv(join('..', 'subjects.csv'))
    subjects = subjects[~((subjects['expression'] == 0) & (subjects['sert-cre'] == 1))]
    if behavior:
        subjects = subjects[subjects['include_behavior'] == 1]
    subjects = subjects.reset_index(drop=True)
    return subjects


def paths():
    """
    Make a file in the root of the repository called 'serotonin_paths.py' with in it:

    DATA_PATH = '/path/to/Flatiron/data'
    FIG_PATH = '/path/to/save/figures'
    SAVE_PATH = '/path/to/save/data'

    """
    from serotonin_paths import DATA_PATH, FIG_PATH, SAVE_PATH
    return DATA_PATH, FIG_PATH, SAVE_PATH


def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                 "axes.titlesize": 8,
                 "axes.labelsize": 8,
                 "axes.linewidth": 0.5,
                 "lines.linewidth": 1,
                 "lines.markersize": 3,
                 "xtick.labelsize": 7,
                 "ytick.labelsize": 7,
                 "savefig.transparent": True,
                 "xtick.major.size": 2.5,
                 "ytick.major.size": 2.5,
                 "xtick.major.width": 0.5,
                 "ytick.major.width": 0.5,
                 "xtick.minor.size": 2,
                 "ytick.minor.size": 2,
                 "xtick.minor.width": 0.5,
                 "ytick.minor.width": 0.5,
                 'legend.fontsize': 7,
                 'legend.title_fontsize': 7
                 })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    colors = {'sert': sns.color_palette('Dark2')[0],
              'wt': [0.75, 0.75, 0.75],
              'left': sns.color_palette('colorblind')[1],
              'right': sns.color_palette('colorblind')[0],
              'enhanced': sns.color_palette('colorblind')[3],
              'suppressed': sns.color_palette('colorblind')[0],
              'no-modulation': sns.color_palette('colorblind')[7],
              'stim': sns.color_palette('colorblind')[9],
              'no-stim': sns.color_palette('colorblind')[7],
              'probe': sns.color_palette('colorblind')[4],
              'block': sns.color_palette('colorblind')[6],
              'RS': sns.color_palette('Set2')[0],
              'FS': sns.color_palette('Set2')[1]}
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 12
    return colors, dpi


def query_opto_sessions(subject, one=None):
    one = one or ONE()
    sessions = one.alyx.rest('sessions', 'list', subject=subject,
                             task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                             project='serotonin_inference')
    return [sess['url'][-36:] for sess in sessions]


def query_ephys_sessions(selection='aligned', return_subjects=False, one=None):
    if one is None:
        one = ONE()
    if selection == 'all':
        # Query all opto_ephysChoiceWorld sessions
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,serotonin_inference,'
                               'session__qc__lt,50')
    elif selection == 'aligned':
        # Query all ephys-histology aligned sessions
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,serotonin_inference,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0')
    elif selection == 'aligned-behavior':
        # Query sessions with an alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,serotonin_inference,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    else:
        ins = []

    # Only include subjects from subjects.csv
    incl_subjects = load_subjects()
    ins = [i for i in ins if i['session_info']['subject'] in incl_subjects['subject'].values]

    # Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    eids, ind_unique = np.unique(all_eids, return_index=True)
    subjects = all_subjects[ind_unique]
    probes = []
    for i, eid in enumerate(eids):
        probes.append(all_probes[[s == eid for s in all_eids]])
    if return_subjects:
        return eids, probes, subjects
    else:
        return eids, probes


def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False,
                patch_old_opto=True, one=None):
    one = one or ONE()
    data, _ = one.load_datasets(eid, datasets=[
        '_ibl_trials.stimOn_times.npy', '_ibl_trials.feedback_times.npy',
        '_ibl_trials.goCue_times.npy', '_ibl_trials.probabilityLeft.npy',
        '_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy',
        '_ibl_trials.feedbackType.npy', '_ibl_trials.choice.npy',
        '_ibl_trials.firstMovement_times.npy'])
    trials = pd.DataFrame(data=np.vstack(data).T, columns=[
        'stimOn_times', 'feedback_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice', 'firstMovement_times'])
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    if laser_stimulation:
        trials['laser_stimulation'] = one.load_dataset(eid, dataset='_ibl_trials.laserStimulation.npy')
        try:
            trials['laser_probability'] = one.load_dataset(eid, dataset='_ibl_trials.laserProbability.npy')
            trials['probe_trial'] = ((trials['laser_stimulation'] == 0) & (trials['laser_probability'] == 0.75)
                                     | (trials['laser_stimulation'] == 1) & (trials['laser_probability'] == 0.25)).astype(int)
        except:
            trials['laser_probability'] = trials['laser_stimulation'].copy()
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 0), 'laser_probability'] = 0.25
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 1), 'laser_probability'] = 0.75

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

    # Patch datasets that contained a bug in the laser driver code
    ses_date = datetime.strptime(one.get_details(eid)['start_time'][:10], '%Y-%m-%d')
    if ((ses_date < datetime.strptime(DATE_GOOD_OPTO, '%Y-%m-%d'))
            and patch_old_opto and laser_stimulation):

        # The bug flipped laser on and off pulses after a long reaction time trial
        bug_trial = ((trials.loc[trials['laser_stimulation'] == 1, 'feedback_times']
                      - trials.loc[trials['laser_stimulation'] == 1, 'stimOn_times']) > 10).idxmax()
        print(f'Patching buggy opto data, excluding {trials.shape[0] - bug_trial} trials')
        trials = trials[:bug_trial]

    return trials


def remap(ids, source='Allen', dest='Beryl', output='acronym'):
    br = BrainRegions()
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']


def get_full_region_name(acronyms):
    brainregions = BrainRegions()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = brainregions.name[np.argwhere(brainregions.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names


def behavioral_criterion(eids, max_lapse=0.3, max_bias=0.4, min_trials=1, one=None):
    if one is None:
        one = ONE()
    use_eids = []
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
        except Exception:
            print('Could not load session %s' % eid)
    return use_eids


def load_exp_smoothing_trials(eids, stimulated=None, rt_cutoff=0.2, after_probe_trials=0, stim_trial_shift=0,
                              pseudo=False, patch_old_opto=True, min_trials=100, one=None):
    """
    Parameters
    ----------
    eids : list
        List of eids
    stimulated : None or string
        If None, do not return the stimulated array, if a string these are the options:
            all: all laser stimulated trials
            probe: only laser probe trials
            block: only laser block trials (no probes)
            rt: this is a weird one - return a reaction time cut off as stimulated trials
    rt_cutoff : float
        Only used if stimulated = 'rt'. Reaction time cutoff in seconds above which stimulated is 1
    after_probe_trials : int
        Only used if stimulated = 'probe'. How many trials after a probe trial are still counted.
    pseudo : bool
        Whether to use pseudo stimulated blocks or shuffled probes as control
    """

    if isinstance(stimulated, str):
        assert stimulated in ['all', 'probe', 'block', 'rt']

    if one is None:
        one=ONE()
    stimuli_arr, actions_arr, stim_sides_arr, prob_left_arr, stimulated_arr, session_uuids = [], [], [], [], [], []
    for j, eid in enumerate(eids):
        try:
            # Load in trials vectors
            if stimulated is not None and stimulated != 'rt':
                trials = load_trials(eid, invert_stimside=True, laser_stimulation=True,
                                     patch_old_opto=patch_old_opto, one=one)
            else:
                trials = load_trials(eid, invert_stimside=True, laser_stimulation=False,
                                     patch_old_opto=patch_old_opto, one=one)
            if trials.shape[0] < min_trials:
                continue
            if stimulated == 'all':
                stim_trials = trials['laser_stimulation'].values
            elif stimulated == 'probe':
                stim_trials = ((trials['laser_stimulation'] == 1) & (trials['laser_probability'] <= .5)).values
                if pseudo:
                    stim_trials = shuffle(stim_trials)
                for k, ind in enumerate(np.where(stim_trials == 1)[0]):
                    stim_trials[ind:ind + (after_probe_trials + 1)] = 1
            elif stimulated == 'block':
                stim_trials = trials['laser_stimulation'].values
                if 'laser_probability' in trials.columns:
                    stim_trials[(trials['laser_stimulation'] == 0) & (trials['laser_probability'] == .75)] = 1
                    stim_trials[(trials['laser_stimulation'] == 1) & (trials['laser_probability'] == .25)] = 0
            elif stimulated == 'rt':
                stim_trials = (trials['reaction_times'] > rt_cutoff).values
            if stim_trial_shift > 0:
                stim_trials = np.append(np.zeros(stim_trial_shift), stim_trials)[:-stim_trial_shift]
            if stimulated is not None:
                stimulated_arr.append(stim_trials)
            stimuli_arr.append(trials['signed_contrast'].values)
            actions_arr.append(trials['choice'].values)
            stim_sides_arr.append(trials['stim_side'].values)
            prob_left_arr.append(trials['probabilityLeft'].values)
            session_uuids.append(eid)
        except:
            print(f'Could not load trials for {eid}')

    if (len(session_uuids) == 0) and (stimulated is not None and stimulated != 'rt'):
        return [], [], [], [], [], []
    elif len(session_uuids) == 0:
        return [], [], [], [], []

    # Get maximum number of trials across sessions
    max_len = np.array([len(stimuli_arr[k]) for k in range(len(stimuli_arr))]).max()

    # Pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials
    stimuli = np.array([np.concatenate((stimuli_arr[k], np.zeros(max_len-len(stimuli_arr[k]))))
                        for k in range(len(stimuli_arr))])
    actions = np.array([np.concatenate((actions_arr[k], np.zeros(max_len-len(actions_arr[k]))))
                        for k in range(len(actions_arr))])
    prob_left = np.array([np.concatenate((prob_left_arr[k], np.zeros(max_len-len(prob_left_arr[k]))))
                        for k in range(len(prob_left_arr))])
    stim_side = np.array([np.concatenate((stim_sides_arr[k],
                                          np.zeros(max_len-len(stim_sides_arr[k]))))
                          for k in range(len(stim_sides_arr))])
    if stimulated is not None and stimulated != 'rt':
        opto_stim = np.array([np.concatenate((stimulated_arr[k], np.zeros(max_len-len(stimulated_arr[k]))))
                              for k in range(len(stimulated_arr))])
    session_uuids = np.array(session_uuids)

    if session_uuids.shape[0] == 1:
        stimuli = np.array([np.squeeze(stimuli)])
        actions = np.array([np.squeeze(actions)])
        prob_left = np.array([np.squeeze(prob_left)])
        stim_side = np.array([np.squeeze(stim_side)])
        if stimulated is not None and stimulated != 'rt':
            opto_stim = np.array([np.squeeze(opto_stim)])

    if stimulated is not None and stimulated != 'rt':
        return actions, stimuli, stim_side, prob_left, opto_stim, session_uuids
    else:
        return actions, stimuli, stim_side, prob_left, session_uuids


def load_opto_times(eid, one=None):
    if one is None:
        one = ONE()

    # Load in laser pulses
    try:
        one.load_datasets(eid, datasets=[
            '_spikeglx_ephysData_g0_t0.nidq.cbin', '_spikeglx_ephysData_g0_t0.nidq.meta',
            '_spikeglx_ephysData_g0_t0.nidq.ch'], download_only=True)
    except:
        one.load_datasets(eid, datasets=[
            '_spikeglx_ephysData_g1_t0.nidq.cbin', '_spikeglx_ephysData_g1_t0.nidq.meta',
            '_spikeglx_ephysData_g1_t0.nidq.ch'], download_only=True)
    session_path = one.eid2path(eid)
    nidq_file = glob(str(session_path.joinpath('raw_ephys_data/_spikeglx_ephysData_g*_t0.nidq.cbin')))[0]
    sr = spikeglx.Reader(nidq_file)
    offset = int((sr.shape[0] / sr.fs - 1800) * sr.fs)
    opto_trace = sr.read_sync_analog(slice(offset, offset + int(1800 * sr.fs)))[:, 1]
    opto_times = np.arange(offset, offset + len(opto_trace)) / sr.fs

    # Get start times of pulse trains
    opto_high_times = opto_times[opto_trace > 1]
    if len(opto_high_times) == 0:
        print(f'No pulses found for {eid}')
        return []
    else:
        opto_train_times = opto_high_times[np.concatenate(([True], np.diff(opto_high_times) > 1))]
        assert np.sum(np.diff(opto_train_times) > 300) == 1, 'Could not find passive laser pulses'
        opto_train_times = opto_train_times[np.where(np.diff(opto_train_times) > 300)[0][0]+1:]
        return opto_train_times


def query_bwm_sessions(selection='all', return_subjects=False, one=None):
    if one is None:
        one = ONE()
    if selection == 'all':
        # Query all ephysChoiceWorld sessions
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50')
    elif selection == 'aligned':
        # Query all sessions with at least one alignment
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0')
    elif selection == 'resolved':
        # Query all sessions with resolved alignment
         ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_resolved,True')
    elif selection == 'aligned-behavior':
        # Query sessions with at least one alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    elif selection == 'resolved-behavior':
        # Query sessions with resolved alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_resolved,True,'
                               'session__extended_qc__behavior,1')
    else:
        ins = []

    # Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    eids, ind_unique = np.unique(all_eids, return_index=True)
    subjects = all_subjects[ind_unique]
    probes = []
    for i, eid in enumerate(eids):
        probes.append(all_probes[[s == eid for s in all_eids]])
    if return_subjects:
        return eids, probes, subjects
    else:
        return eids, probes


def fit_psychfunc(stim_levels, n_trials, proportion):
    # Fit a psychometric function with two lapse rates
    #
    # Returns vector pars with [bias, threshold, lapselow, lapsehigh]
    import psychofit as psy
    assert(stim_levels.shape == n_trials.shape == proportion.shape)
    if stim_levels.max() <= 1:
        stim_levels = stim_levels * 100

    pars, _ = psy.mle_fit_psycho(np.vstack((stim_levels, n_trials, proportion)),
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array([0, 20, 0.05, 0.05]),
                                 parmin=np.array([-100, 5, 0, 0]),
                                 parmax=np.array([100, 100, 1, 1]))
    return pars


def plot_psychometric(trials, ax, **kwargs):
    import psychofit as psy
    if trials['signed_contrast'].max() <= 1:
        trials['signed_contrast'] = trials['signed_contrast'] * 100

    stim_levels = np.sort(trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, trials.groupby('signed_contrast').size(),
                         trials.groupby('signed_contrast').mean()['right_choice'])

    # plot psychfunc
    sns.lineplot(x=np.arange(-27, 27), y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)),
                 ax=ax, **kwargs)

    # plot psychfunc: -100, +100
    sns.lineplot(x=np.arange(-36, -31), y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)),
                 ax=ax, **kwargs)
    sns.lineplot(x=np.arange(31, 36), y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)),
                 ax=ax, **kwargs)

    # now break the x-axis
    trials['signed_contrast'].replace(-100, -35)
    trials['signed_contrast'].replace(100, 35)

    # plot datapoints with errorbars on top
    sns.lineplot(x=trials['signed_contrast'], y=trials['right_choice'], ax=ax,
                     **{**{'err_style':"bars",
                     'linewidth':0, 'linestyle':'None', 'mew':0.5,
                     'marker':'o', 'ci':68}, **kwargs})

    ax.set(xticks=[-35, -25, -12.5, 0, 12.5, 25, 35], xlim=[-40, 40], ylim=[0, 1.02],
           yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=['0', '25', '50', '75', '100'],
           ylabel='Right choices', xlabel='Contrast (%)')
    ax.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'])
    #break_xaxis()


def break_xaxis(y=-0.004, **kwargs):

    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')


def get_bias(trials):
    import psychofit as psy
    """
    Calculate bias by fitting psychometric curves to the 80/20 and 20/80 blocks, finding the
    point on the y-axis when contrast = 0% and getting the difference.
    """
    if len(trials) == 0:
        return np.nan

    # 20/80 blocks
    these_trials = trials[trials['probabilityLeft'] == 0.2]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_right = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                               these_trials.groupby('signed_contrast').mean()['right_choice'])
    bias_right = psy.erf_psycho_2gammas(pars_right, 0)

    # 80/20 blocks
    these_trials = trials[trials['probabilityLeft'] == 0.8]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_left = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                              these_trials.groupby('signed_contrast').mean()['right_choice'])
    bias_left = psy.erf_psycho_2gammas(pars_left, 0)

    return bias_right - bias_left


def fit_glm(behav, prior_blocks=True, opto_stim=False, folds=5):

    # drop trials with contrast-level 50, only rarely present (should not be its own regressor)
    behav = behav[np.abs(behav.signed_contrast) != 50]

    # add extra parameters to GLM
    model_str = 'choice ~ 1 + stimulus_side:C(contrast, Treatment) + previous_choice:C(previous_outcome)'
    if opto_stim:
        model_str = model_str + ' + laser_stimulation'
    if prior_blocks:
        model_str = model_str + ' + block_id'

    # use patsy to easily build design matrix
    endog, exog = patsy.dmatrices(model_str, data=behav.dropna(subset=['trial_feedback_type', 'choice', 'previous_choice', 'previous_outcome']).reset_index(),
                                  return_type='dataframe')

    # remove the one column (with 0 contrast) that has no variance
    if 'stimulus_side:C(contrast, Treatment)[0.0]' in exog.columns:
        exog.drop(columns=['stimulus_side:C(contrast, Treatment)[0.0]'], inplace=True)

    # recode choices for logistic regression
    endog['choice'] = endog['choice'].map({-1:0, 1:1})

    # rename columns
    exog.rename(columns={'Intercept': 'bias',
              'stimulus_side:C(contrast, Treatment)[6.25]': '6.25',
             'stimulus_side:C(contrast, Treatment)[12.5]': '12.5',
             'stimulus_side:C(contrast, Treatment)[25.0]': '25',
             'stimulus_side:C(contrast, Treatment)[50.0]': '50',
             'stimulus_side:C(contrast, Treatment)[100.0]': '100',
             'previous_choice:C(previous_outcome)[-1.0]': 'unrewarded',
             'previous_choice:C(previous_outcome)[1.0]': 'rewarded'},
             inplace=True)

    # NOW FIT THIS WITH STATSMODELS - ignore NaN choices
    logit_model = sm.Logit(endog, exog)
    res = logit_model.fit_regularized(disp=False) # run silently

    # what do we want to keep?
    params = pd.DataFrame(res.params).T
    params['pseudo_rsq'] = res.prsquared # https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.prsquared.html?highlight=pseudo
    params['condition_number'] = np.linalg.cond(exog)

    # ===================================== #
    # ADD MODEL ACCURACY - cross-validate

    kf = KFold(n_splits=folds, shuffle=True)
    acc = np.array([])
    for train, test in kf.split(endog):
        X_train, X_test, y_train, y_test = exog.loc[train], exog.loc[test], \
                                           endog.loc[train], endog.loc[test]
        # fit again
        logit_model = sm.Logit(y_train, X_train)
        res = logit_model.fit_regularized(disp=False)  # run silently

        # compute the accuracy on held-out data [from Luigi]:
        # suppose you are predicting Pr(Left), let's call it p,
        # the % match is p if the actual choice is left, or 1-p if the actual choice is right
        # if you were to simulate it, in the end you would get these numbers
        y_test['pred'] = res.predict(X_test)
        y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
        acc = np.append(acc, y_test['pred'].mean())

    # average prediction accuracy over the K folds
    params['accuracy'] = np.mean(acc)

    return params  # wide df