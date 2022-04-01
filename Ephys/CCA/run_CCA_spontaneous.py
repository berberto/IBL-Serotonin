#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.singlecell import calculate_peths
from brainbox.io.one import SpikeSortingLoader
from scipy.stats import mannwhitneyu, sem
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, figure_style)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
cca = CCA(n_components=1)
pca = PCA(n_components=10)

# Settings
NEURON_QC = True
MIN_NEURONS = 10  # per region
N_SPONT = 500  # number of trials in the spontaneous activity
N_PERMUT = 500  # number of times to get spontaneous population correlation for permutation testing
WIN_SIZE = 0.2  # window size in seconds
PRE_TIME = 1
POST_TIME = 2
SMOOTHING = 0
MIN_FR = 0.5  # minimum firing rate over the whole recording
N_PC = 10  # number of PCs to use
PLOT_IND = True  # plot individual region pairs
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA')
np.random.seed(42)  # fix random seed for reproducibility

# Query sessions
rec = query_ephys_sessions(one=one)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

cca_df = pd.DataFrame()
for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
    date = rec.loc[rec['eid'] == eid, 'date'].values[0]
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

    # Load in neural data of both probes of the recording
    spikes, clusters, channels, clusters_pass = dict(), dict(), dict(), dict()
    for (pid, probe) in zip(rec.loc[rec['eid'] == eid, 'pid'].values, rec.loc[rec['eid'] == eid, 'probe'].values):

        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes[probe], clusters[probe], channels[probe] = sl.load_spike_sorting()
        clusters[probe] = sl.merge_clusters(spikes[probe], clusters[probe], channels[probe])

        # Filter neurons that pass QC and artifact neurons
        if NEURON_QC:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass[probe] = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass[probe] = np.unique(spikes.clusters)
        clusters_pass[probe] = clusters_pass[probe][~np.isin(clusters_pass[probe], artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        clusters[probe]['region'] = remap(clusters[probe]['atlas_id'], combine=True)

    # Create population activity arrays for all regions
    pca_spont, pca_opto = dict(), dict()
    for probe in spikes.keys():
        for region in np.unique(clusters[probe]['region']):

             # Exclude neurons with low firing rates
             clusters_in_region = np.where(clusters[probe]['region'] == region)[0]
             fr = np.empty(clusters_in_region.shape[0])
             for nn, neuron_id in enumerate(clusters_in_region):
                 fr[nn] = np.sum(spikes[probe].clusters == neuron_id) / spikes[probe].clusters[-1]
             clusters_in_region = clusters_in_region[fr >= MIN_FR]

             # Get spikes and clusters
             spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)
                                               & np.isin(spikes[probe].clusters, clusters_pass[probe])]
             clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)
                                                  & np.isin(spikes[probe].clusters, clusters_pass[probe])]

             if (len(np.unique(clus_region)) >= MIN_NEURONS) & (region != 'root'):
                 print(f'Loading population activity for {region}')

                 # Get binned spikes for SPONTANEOUS activity and subtract mean
                 pop_spont_pca = np.empty([N_PERMUT, N_SPONT, N_PC])
                 for jj in range(N_PERMUT):

                     # Get random times for spontaneous activity
                     spont_on_times = np.sort(np.random.uniform(
                         opto_train_times[0] - (6 * 60), opto_train_times[0], size=N_SPONT))
                     spont_times = np.column_stack((spont_on_times, spont_on_times + WIN_SIZE))

                     # Get population activity
                     this_pop_spont, _ = get_spike_counts_in_bins(spks_region, clus_region, spont_times)
                     this_pop_spont = this_pop_spont.T
                     this_pop_spont = this_pop_spont - np.mean(this_pop_spont, axis=0)
                     pop_spont_pca[jj, :, :] = pca.fit_transform(this_pop_spont)

                 # Get PSTH and binned spikes for OPTO activity
                 psth_opto, binned_spks_opto = calculate_peths(
                     spks_region, clus_region, np.unique(clus_region), opto_train_times, pre_time=PRE_TIME,
                     post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING, return_fr=False)

                 # Subtract mean PSTH from each opto stim
                 for tt in range(binned_spks_opto.shape[0]):
                     binned_spks_opto[tt, :, :] = binned_spks_opto[tt, :, :] - psth_opto['means']

                 # Perform PCA
                 pca_spont[region] = pop_spont_pca
                 pca_opto[region] = np.empty([binned_spks_opto.shape[0], N_PC, binned_spks_opto.shape[2]])
                 for tb in range(binned_spks_opto.shape[2]):
                     pca_opto[region][:, :, tb] = pca.fit_transform(binned_spks_opto[:, :, tb])

    # Perform CCA per region pair
    print('Starting CCA per region pair')
    all_cca_df = pd.DataFrame()
    for region_1 in pca_spont.keys():
        for region_2 in pca_spont.keys():
            if region_1 == region_2:
                continue
            print(f'Calculating {region_1}-{region_2}')

            # Fit communication subspace axis and get population correlation during
            # spontaneous activity using 80/20 cross-validation
            r_spont = np.empty(N_PERMUT)
            r_opto = np.empty([N_PERMUT, pca_opto[region_1].shape[2]])
            for jj in range(N_PERMUT):
                x_train, x_test, y_train, y_test = train_test_split(
                    pca_spont[region_1][jj, :, :], pca_spont[region_2][jj, :, :], test_size=0.2,
                    random_state=42, shuffle=True)
                cca.fit(x_train, y_train)
                spont_x, spont_y = cca.transform(x_test, y_test)
                r_spont[jj], _ = pearsonr(np.squeeze(spont_x), np.squeeze(spont_y))

                # Use fitted CCA axis to get population correlation during opto stimulus per time bin
                r_this_opto = np.empty(pca_opto[region_1].shape[2])
                for tb in range(pca_opto[region_1].shape[2]):
                    opto_x, opto_y = cca.transform(pca_opto[region_1][:, :, tb],
                                                   pca_opto[region_2][:, :, tb])
                    r_this_opto[tb], _ = pearsonr(np.squeeze(opto_x), np.squeeze(opto_y))

                # Add to df and matrix
                r_opto[jj, :] = r_this_opto
                """
                all_cca_df = pd.concat((all_cca_df, pd.DataFrame(data={
                    'subject': subject, 'date': date, 'eid': eid, 'region_1': region_1, 'region_2': region_2,
                    'region_pair': f'{region_1}-{region_2}', 'r_spont': r_spont[jj], 'r_opto': r_this_opto,
                    'time': psth_opto['tscale'], 'it': jj})), ignore_index=True)
                """

            # Plot this region pair
            if PLOT_IND:
                colors, dpi = figure_style()
                f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
                ax1.fill_between([psth_opto['tscale'][0], psth_opto['tscale'][-1]],
                                 [np.mean(r_spont) - np.std(r_spont), np.mean(r_spont) - np.std(r_spont)],
                                 [np.mean(r_spont) + np.std(r_spont), np.mean(r_spont) + np.std(r_spont)],
                                 color='grey', alpha=0.2, lw=0)
                #ax1.fill_between([psth_opto['tscale'][0], psth_opto['tscale'][-1]],
                #                 [np.quantile(r_spont, 0.05), np.quantile(r_spont, 0.05)],
                #                 [np.quantile(r_spont, 0.95), np.quantile(r_spont, 0.95)],
                #                 color='grey', alpha=0.2, lw=0)
                ax1.fill_between(psth_opto['tscale'], np.mean(r_opto, axis=0)-np.std(r_opto)/2,
                                 np.mean(r_opto, axis=0)+np.std(r_opto)/2, alpha=0.2)
                #ax1.fill_between(psth_opto['tscale'], np.quantile(r_opto, 0.05, axis=0),
                #                 np.quantile(r_opto, 0.95, axis=0), alpha=0.2)
                ax1.plot(psth_opto['tscale'], np.mean(r_opto, axis=0), lw=2)
                ax1.plot([0, 0], ax1.get_ylim(), color='k', ls='--')
                ax1.set(xlabel='Time (s)', ylabel='Population correlation (r)',
                        title=f'{region_1}-{region_2}')
                sns.despine(trim=True)
                plt.tight_layout()
                plt.savefig(join(fig_path, 'RegionPairsStd', f'{region_1}-{region_2}_{subject}_{date}'), dpi=300)
                plt.close(f)

"""
cca_df['r_subtract'] = cca_df['r_opto'] - cca_df['r_spont']
sns.lineplot(x='time', y='r_opto', data=cca_df, hue='region_pair', ci=68)

# Save results
cca_df.to_csv(join(save_path, 'cca_results.csv'))
"""





