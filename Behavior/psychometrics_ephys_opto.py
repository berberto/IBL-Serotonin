#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (load_trials, plot_psychometric, paths, behavioral_criterion,
                                 fit_psychfunc, figure_style, load_subjects)
from one.api import ONE
one = ONE()

# Settings
PLOT_SINGLE_ANIMALS = True
fig_path, _ = paths()
fig_path = join(fig_path, 'Behavior', 'Ephys')
subjects = load_subjects()

#subjects = subjects[subjects['subject'] == 'ZFM-01867'].reset_index(drop=True)

bias_df, lapse_df, psy_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_ephysChoiceWorld')

    # Apply behavioral critria
    eids = behavioral_criterion(eids)

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
            these_trials['session'] = ses_count
            trials = trials.append(these_trials, ignore_index=True)
            ses_count = ses_count + 1
        except:
            pass
    if len(trials) == 0:
        continue
    if 'laser_probability' not in trials.columns:
        trials['laser_probability'] = trials['laser_stimulation'].copy()

    # Get bias shift
    bias_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] == 0.25)].mean()
                          - trials[(trials['probabilityLeft'] == 0.2)
                                   & (trials['laser_stimulation'] == 0)
                                   & (trials['laser_probability'] == 0.25)].mean())['right_choice']
    bias_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                              & (trials['laser_stimulation'] == 1)
                              & (trials['laser_probability'] == 0.75)].mean()
                       - trials[(trials['probabilityLeft'] == 0.2)
                                & (trials['laser_stimulation'] == 1)
                                & (trials['laser_probability'] == 0.75)].mean())['right_choice']
    bias_catch_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                    & (trials['laser_probability'] == 0.25)].mean()
                             - trials[(trials['probabilityLeft'] == 0.2)
                                      & (trials['laser_stimulation'] == 1)
                                      & (trials['laser_probability'] == 0.25)].mean())['right_choice']
    bias_catch_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                       & (trials['laser_stimulation'] == 0)
                                       & (trials['laser_probability'] == 0.75)].mean()
                                - trials[(trials['probabilityLeft'] == 0.2)
                                         & (trials['laser_stimulation'] == 0)
                                         & (trials['laser_probability'] == 0.75)].mean())['right_choice']
    # Get RT
    rt_no_stim = trials[(trials['laser_stimulation'] == 0)
                        & (trials['laser_probability'] == 0.25)].median()['reaction_times']
    rt_stim = trials[(trials['laser_stimulation'] == 1)
                     & (trials['laser_probability'] == 0.75)].median()['reaction_times']
    rt_catch_no_stim = trials[(trials['laser_stimulation'] == 0)
                              & (trials['laser_probability'] == 0.75)].median()['reaction_times']
    rt_catch_stim = trials[(trials['laser_stimulation'] == 1)
                           & (trials['laser_probability'] == 0.25)].median()['reaction_times']
    bias_df = bias_df.append(pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'expression': subjects.loc[i, 'expression'],
        'bias': [bias_no_stim, bias_stim, bias_catch_stim, bias_catch_no_stim],
        'rt': [rt_no_stim, rt_stim, rt_catch_stim, rt_catch_no_stim],
        'opto_stim': [0, 1, 1, 0], 'catch_trial': [0, 0, 1, 1]}))

    # Get lapse rates
    lapse_l_l_ns = trials.loc[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == -1)
                              & (trials['laser_probability'] == 0), 'correct'].mean()
    lapse_r_l_ns = trials.loc[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 1)
                              & (trials['laser_probability'] == 0), 'correct'].mean()
    lapse_l_r_ns = trials.loc[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == -1)
                              & (trials['laser_probability'] == 0), 'correct'].mean()
    lapse_r_r_ns = trials.loc[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 1)
                              & (trials['laser_probability'] == 0), 'correct'].mean()
    lapse_l_l_s = trials.loc[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == -1)
                              & (trials['laser_probability'] == 1), 'correct'].mean()
    lapse_r_l_s = trials.loc[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 1)
                              & (trials['laser_probability'] == 1), 'correct'].mean()
    lapse_l_r_s = trials.loc[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == -1)
                              & (trials['laser_probability'] == 1), 'correct'].mean()
    lapse_r_r_s = trials.loc[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 1)
                              & (trials['laser_probability'] == 1), 'correct'].mean()
    lapse_df = lapse_df.append(pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'expression': subjects.loc[i, 'expression'],
        'lapse': [lapse_l_l_ns, lapse_r_l_ns, lapse_l_r_ns, lapse_r_r_ns,
                  lapse_l_l_s, lapse_r_l_s, lapse_l_r_s, lapse_r_r_s],
        'opto_stim': [0, 0, 0, 0, 1, 1, 1, 1],
        'stim_side': ['l', 'r', 'l', 'r', 'l', 'r', 'l', 'r'],
        'bias_side': ['l', 'l', 'r', 'r', 'l', 'l', 'r', 'r']}))

    # Get fit parameters
    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 0)
                          & (trials['laser_probability'] != 0.75)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.8,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))
    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 0)
                          & (trials['laser_probability'] != 0.75)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.2,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))
    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 1)
                          & (trials['laser_probability'] != 0.25)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 1, 'prob_left': 0.8,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))
    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 1)
                          & (trials['laser_probability'] != 0.25)]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'expression': subjects.loc[i, 'expression'],
        'opto_stim': 1, 'prob_left': 0.2,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    # Plot
    if PLOT_SINGLE_ANIMALS:
        colors, dpi = figure_style()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi, sharey=True)

        # plot_psychometric(trials[trials['probabilityLeft'] == 0.5], ax=ax1, color='k')
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] != 0.75)], ax=ax1, color=colors['left'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['laser_probability'] != 0.25)], ax=ax1,
                          color=colors['left'], linestyle='--')
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] != 0.75)], ax=ax1, color=colors['right'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['laser_probability'] != 0.25)], ax=ax1,
                          color=colors['right'], linestyle='--')
        ax1.text(-25, 0.75, '20:80', color=colors['right'])
        ax1.text(25, 0.25, '80:20', color=colors['left'])
        ax1.set(title='dashed line = opto stim')

        catch_trials = trials[((trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0))
                              | ((trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1))]

        ax2.errorbar([0, 1],
                     [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].mean(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].mean()],
                     [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].sem(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].sem()],
                     marker='o', label='Stim', color=colors['right'], ls='--')
        ax2.errorbar([0, 1],
                     [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].mean(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].mean()],
                     [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].sem(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].sem()],
                     marker='o', label='No stim', color=colors['right'])
        ax2.errorbar([0, 1],
                     [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].mean(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].mean()],
                     [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].sem(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].sem()],
                     marker='o', label='Stim', color=colors['left'], ls='--')
        ax2.errorbar([0, 1],
                     [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].mean(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].mean()],
                     [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                             & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].sem(),
                      catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].sem()],
                     marker='o', label='No stim', color=colors['left'])
        ax2.set(xticks=[0, 1], xticklabels=['Normal trials', 'Catch trials'], title='0% contrast trials')

        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, '%s_opto_behavior_psycurve' % nickname))

# %% Plot
psy_avg_block_df = psy_df.groupby(['subject', 'opto_stim']).mean()
psy_avg_block_df['lapse_both'] = psy_avg_block_df.loc[:, 'lapse_l':'lapse_r'].mean(axis=1)
psy_avg_block_df = psy_avg_block_df.reset_index()
colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6, 3.5), dpi=dpi)
for i, subject in enumerate(bias_df['subject']):
    ax1.plot([1, 2], bias_df.loc[(bias_df['subject'] == subject) & (bias_df['catch_trial'] == 0), 'bias'],
             color = colors[bias_df.loc[bias_df['subject'] == subject, 'expression'].unique()[0]],
             marker='o', ms=2)
ax1.set(xlabel='', xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylabel='Bias', ylim=[0, 0.7])

for i, subject in enumerate(bias_df['subject']):
    ax2.plot([1, 2], bias_df.loc[(bias_df['subject'] == subject) & (bias_df['catch_trial'] == 1), 'bias'],
             color = colors[bias_df.loc[bias_df['subject'] == subject, 'expression'].unique()[0]],
             marker='o', ms=2)
ax2.set(xlabel='', xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylabel='Bias', ylim=[0, 0.7],
        title='Catch trials')

delta_block = (bias_df.loc[(bias_df['opto_stim'] == 1) & (bias_df['catch_trial'] == 0), 'bias'].values -
               bias_df.loc[(bias_df['opto_stim'] == 0) & (bias_df['catch_trial'] == 0), 'bias'].values)
delta_probe = (bias_df.loc[(bias_df['opto_stim'] == 1) & (bias_df['catch_trial'] == 1), 'bias'].values -
               bias_df.loc[(bias_df['opto_stim'] == 0) & (bias_df['catch_trial'] == 1), 'bias'].values)
sert_cre = bias_df.loc[(bias_df['opto_stim'] == 1) & (bias_df['catch_trial'] == 0), 'sert-cre'].values
ax3.plot([0, 0], [-0.2, 0.2], ls='--', color='gray')
ax3.plot([-0.2, 0.2], [0, 0], ls='--', color='gray')
ax3.scatter(delta_block[sert_cre == 1], delta_probe[sert_cre ==1], color=colors[0])
ax3.scatter(delta_block[sert_cre == 0], delta_probe[sert_cre ==0], color=colors[1])
ax3.set(xlim=(-0.2, 0.2), ylim=(-0.2, 0.2), xlabel='Bias change block trials', ylabel='Bias change probe trials')

sns.lineplot(x='opto_stim', y='threshold', hue='sert-cre', style='subject', estimator=None,
             data=psy_df.groupby(['subject', 'opto_stim']).mean(), dashes=False,
             markers=['o']*int(bias_df.shape[0]/4), palette=colors, hue_order=[1, 0],
             legend=False, lw=1, ms=4, ax=ax4)
ax4.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Threshold')

for i, subject in enumerate(psy_avg_block_df['subject']):
    ax5.plot([1, 2], psy_avg_block_df.loc[(psy_avg_block_df['subject'] == subject), 'lapse_both'],
             color = colors[int(psy_avg_block_df.loc[psy_avg_block_df['subject'] == subject, 'expression'].unique()[1])],
             marker='o', ms=2)
ax5.set(xlabel='', xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylabel='Lapse rate')

for i, subject in enumerate(bias_df['subject']):
    ax6.plot([1, 2], bias_df.loc[(bias_df['subject'] == subject) & (bias_df['catch_trial'] == 0), 'rt'],
             color = colors[bias_df.loc[bias_df['subject'] == subject, 'expression'].unique()[0]],
             marker='o', ms=2)
ax6.set(xlabel='', xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylabel='Median reaction time')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'summary_psycurve'))

