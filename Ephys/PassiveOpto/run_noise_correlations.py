#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from brainbox.population.decode import get_spike_counts_in_bins
import pandas as pd
import seaborn as sns
from os import mkdir
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from brainbox.task.closed_loop import roc_single_event
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas, BrainRegions
ba = AllenAtlas()
br = BrainRegions()
one = ONE()

# Settings
MIN_NEURONS = 5
MIN_FR = 0.01  # spks/s
BASELINE = [-0.5, 0]
STIM = [0.5, 1]
LABEL = 'late'
PLOT = True
NEURON_QC = True
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'Correlations')

# Query sessions
eids, _, subjects = query_ephys_sessions(return_subjects=True, one=one)

# Get artifact neurons
artifact_neurons = get_artifact_neurons()

#eids = [eids[3]]

# %%
corr_df = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

    # Initialize variables
    pop_act_bl, pop_act_stim = [], []

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Filter neurons that pass QC
        if NEURON_QC:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass = np.unique(spikes[probe].clusters)
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        if len(spikes[probe].clusters) == 0:
            continue

        # Exclude artifact neurons
        clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
            (artifact_neurons['eid'] == eid) & (artifact_neurons['probe'] == probe), 'neuron_id'].values])
        if clusters_pass.shape[0] == 0:
            continue

        # Get merged regions
        clusters[probe]['region'] = remap(clusters[probe]['atlas_id'], combine=True, split_thalamus=True)
        clusters_regions = clusters[probe]['region'][clusters_pass]

        # Loop over regions
        for r, region in enumerate(np.unique(clusters_regions)):
            if region == 'root':
                continue
            print(region)

            # Select spikes and clusters in this brain region
            clusters_in_region = clusters_pass[clusters_regions == region]
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]
            if len(clusters_in_region) < MIN_NEURONS:
                continue

            # Get baseline activity over all neurons
            times = np.column_stack((((opto_train_times + BASELINE[0]),
                                     ((opto_train_times + BASELINE[1])))))
            pop_vector, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
            pop_vector = pop_vector.T
            if len(pop_act_bl) == 0:
                pop_act_bl = pop_vector
                pop_regions = np.array([region] * pop_vector.shape[1])
            else:
                pop_act_bl = np.hstack((pop_act_bl, pop_vector))
                pop_regions = np.concatenate((pop_regions, np.array([region] * pop_vector.shape[1])))

            # Get stim activity over all neurons
            times = np.column_stack((((opto_train_times + STIM[0]),
                                     ((opto_train_times + STIM[1])))))
            pop_vector, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
            pop_vector = pop_vector.T
            if len(pop_act_stim) == 0:
                pop_act_stim = pop_vector
            else:
                pop_act_stim = np.hstack((pop_act_stim, pop_vector))

    # Remove neurons that do not spike at all
    incl_neurons = (np.sum(pop_act_bl, axis=0) > 0) & (np.sum(pop_act_stim, axis=0) > 0)
    pop_act_bl = pop_act_bl[:, incl_neurons]
    pop_act_stim = pop_act_stim[:, incl_neurons]
    pop_regions = pop_regions[incl_neurons]

    # Calculate noise correlations for baseline
    corr_mat_bl = np.corrcoef(pop_act_bl, rowvar=False)
    mean_corr_arr = corr_mat_bl[np.triu_indices(corr_mat_bl.shape[0])]
    mean_corr_arr = mean_corr_arr[mean_corr_arr != 1]
    mean_corr_bl = np.mean(mean_corr_arr)

    # Calculate noise correlations for stim
    corr_mat_stim = np.corrcoef(pop_act_stim, rowvar=False)
    mean_corr_arr = corr_mat_stim[np.triu_indices(corr_mat_stim.shape[0])]
    mean_corr_arr = mean_corr_arr[mean_corr_arr != 1]
    mean_corr_stim = np.mean(mean_corr_arr)

    # Get change in correlation
    corr_mat_change = corr_mat_stim - corr_mat_bl

    # Get change in correlation per region into dataframe
    for r, region in enumerate(np.unique(pop_regions)):
        reg_change = corr_mat_change[np.ix_(pop_regions == region, pop_regions == region)]
        corr_change = np.mean(reg_change[np.triu_indices(reg_change.shape[0], k=1)])
        corr_df = pd.concat((corr_df, pd.DataFrame(index=[corr_df.shape[0] + 1], data={
            'corr_change': corr_change, 'region': region, 'subject': subject, 'date': date})))

    # Plot result
    plt_regions, plt_reg_ind, plt_reg_loc = np.unique(pop_regions, return_counts=True, return_index=True)
    plt_regions = plt_regions[np.argsort(plt_reg_ind)]
    plt_reg_loc = plt_reg_loc[np.argsort(plt_reg_ind)]

    colors, dpi = figure_style()
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), dpi=dpi)

    corr_mat_bl[np.diag_indices_from(corr_mat_bl)] = 0
    ax1.imshow(corr_mat_bl, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    for l, loc in enumerate(np.cumsum(plt_reg_loc)[:-1]):
        ax1.plot([0, corr_mat_bl.shape[1]-1], [loc, loc], color='k', ls='--')
        ax1.plot([loc, loc], [0, corr_mat_bl.shape[1]-1], color='k', ls='--')
    ax1.set(xticks=np.cumsum(plt_reg_loc) - (plt_reg_loc / 2),
            xticklabels=plt_regions,
            yticks=np.cumsum(plt_reg_loc) - (plt_reg_loc / 2),
            yticklabels=plt_regions)
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_title('Baseline', fontweight='bold')
    for key, spine in ax1.spines.items():
        spine.set_visible(False)
    ax1.tick_params(axis=u'both', which=u'both',length=0)

    corr_mat_stim[np.diag_indices_from(corr_mat_stim)] = 0
    ax2.imshow(corr_mat_stim, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    for l, loc in enumerate(np.cumsum(plt_reg_loc)[:-1]):
        ax2.plot([0, corr_mat_bl.shape[1]-1], [loc, loc], color='k', ls='--')
        ax2.plot([loc, loc], [0, corr_mat_bl.shape[1]-1], color='k', ls='--')
    ax2.set(xticks=np.cumsum(plt_reg_loc) - (plt_reg_loc / 2),
            xticklabels=plt_regions, yticks=[])
    ax2.set_title('5-HT stimulation', fontweight='bold')
    for key, spine in ax2.spines.items():
        spine.set_visible(False)
    ax2.tick_params(axis=u'both', which=u'both',length=0)
    ax2.tick_params(axis='x', labelrotation=90)

    axin = inset_axes(ax3, width="5%", height="100%", loc='lower left',
                      bbox_to_anchor=(1.05, 0, 1, 1),
                      bbox_transform=ax3.transAxes, borderpad=0)
    img = ax3.imshow(corr_mat_stim - corr_mat_bl, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    for l, loc in enumerate(np.cumsum(plt_reg_loc)[:-1]):
        ax3.plot([0, corr_mat_bl.shape[1]-1], [loc, loc], color='k', ls='--')
        ax3.plot([loc, loc], [0, corr_mat_bl.shape[1]-1], color='k', ls='--')
    ax3.set(xticks=np.cumsum(plt_reg_loc) - (plt_reg_loc / 2),
            xticklabels=plt_regions, yticks=[])
    ax3.set_title('Stim-Baseline', fontweight='bold')
    for key, spine in ax3.spines.items():
        spine.set_visible(False)
    ax3.tick_params(axis=u'both', which=u'both',length=0)
    ax3.tick_params(axis='x', labelrotation=90)

    cbar = f.colorbar(img, cax=axin)
    cbar.ax.set_ylabel('Correlation (r)', rotation=270, labelpad=10)
    cbar.ax.set_yticks(np.arange(-0.5, 0.6, 0.25))

    plt.savefig(join(fig_path, f'{subject}_{date}_{LABEL}.jpg'), dpi=300)
    plt.savefig(join(fig_path, f'{subject}_{date}_{LABEL}.pdf'))

    plt.close(f)

corr_df.to_csv(join(save_path, f'correlations_{LABEL}.csv'), index=False)

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(4, 4), dpi=dpi)
sns.swarmplot(x='region', y='corr_change', data=corr_df, ax=ax1)