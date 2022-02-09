#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.singlecell import calculate_peths
from serotonin_functions import paths, load_passive_opto_times, combine_regions, load_subjects
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.01
SMOOTHING = 0.05
BASELINE = [-1, 0]
MIN_FR = 0.1
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons')

# Load in light modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons['full_region'] = combine_regions(light_neurons['region'], split_thalamus=False)
light_neurons = light_neurons[light_neurons['full_region'] != 'root']

# Only select neurons from sert-cre mice
subjects = load_subjects()
light_neurons = light_neurons[light_neurons['subject'].isin(
    subjects.loc[subjects['sert-cre'] == 1, 'subject'].values)]

# %% Loop over sessions
peths_df = pd.DataFrame()
for i, eid in enumerate(np.unique(light_neurons['eid'])):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_train_times) == 0:
        continue

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

    # Loop over probes
    for p, probe in enumerate(spikes.keys()):

        # Take slice of dataframe
        these_neurons = light_neurons[(light_neurons['modulated'] == 1)
                                      & (light_neurons['eid'] == eid)
                                      & (light_neurons['probe'] == probe)]

        # Get peri-event time histogram
        peths, _ = calculate_peths(spikes[probe].times, spikes[probe].clusters,
                                   these_neurons['neuron_id'].values,
                                   opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
        tscale = peths['tscale']

        # Loop over neurons
        for n, index in enumerate(these_neurons.index.values):
            if np.mean(peths['means'][n, :]) > MIN_FR:
                # Calculate percentage change in firing rate
                peth_perc = ((peths['means'][n, :]
                              - np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))]))
                             / np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))])) * 100

                # Calculate percentage change in firing rate
                peth_ratio = ((peths['means'][n, :]
                               - np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))]))
                              / (peths['means'][n, :]
                                 + np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))])))

                # Add to dataframe
                peths_df = pd.concat((peths_df, pd.DataFrame(index=[peths_df.shape[0]], data={
                    'peth': [peths['means'][n, :]], 'peth_perc': [peth_perc], 'peth_ratio': [peth_ratio],
                    'region': these_neurons.loc[index, 'full_region'], 'modulation': these_neurons.loc[index, 'roc_auc'],
                    'neuron_id': these_neurons.loc[index, 'neuron_id'], 'subject': these_neurons.loc[index, 'subject'],
                    'eid': these_neurons.loc[index, 'eid'], 'acronym': these_neurons.loc[index, 'region']})))

# %% Plot

VMIN = -1
VMAX = 1

# Plot all neurons
peths_df = peths_df.sort_values('modulation', ascending=False)
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=dpi)
img = ax1.imshow(np.array(peths_df['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1])
cbar = f.colorbar(img, ax=ax1, shrink=0.7)
cbar.ax.set_ylabel('Ratio change in firing rate', rotation=270, labelpad=10)
ax1.set(xlabel='Time (s)', yticks=[], title=f'All significantly modulated neurons (n={peths_df.shape[0]})')
ax1.plot([0, 0], [-1, 1], ls='--', color='k')

plt.tight_layout()
plt.savefig(join(fig_path, 'all_neurons'), dpi=600)

# Plot per region
peths_df = peths_df.sort_values(['region', 'modulation'], ascending=[True, False])
f, ((ax_th, ax_mpfc, ax_orb, ax_am, ax_ppc, ax_pir),
    (ax_hc, ax_rs, ax_st, ax_sc, ax_zi, _)) = plt.subplots(2, 6, figsize=(9, 4), dpi=dpi)

these_peths = peths_df[peths_df['region'] == 'Medial prefrontal cortex']
img = ax_mpfc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_mpfc.set(xlabel='Time (s)', yticks=[])
ax_mpfc.set_title(f'mPFC\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_mpfc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Orbitofrontal cortex']
img = ax_orb.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_orb.set(xlabel='Time (s)', yticks=[])
ax_orb.set_title(f'Orbitofrontal\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_orb.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Amygdala']
img = ax_am.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_am.set(xlabel='Time (s)', yticks=[])
ax_am.set_title(f'Amygdala\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_am.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Posterior parietal cortex']
img = ax_ppc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_ppc.set(xlabel='Time (s)', yticks=[])
ax_ppc.set_title(f'Pos. par. cortex\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_ppc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Hippocampus']
img = ax_hc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_hc.set(xlabel='Time (s)', yticks=[])
ax_hc.set_title(f'Hippocampus\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_hc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Piriform']
img = ax_pir.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_pir.set(xlabel='Time (s)', yticks=[])
ax_pir.set_title(f'Piriform\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_pir.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Retrosplenial']
img = ax_rs.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_rs.set(xlabel='Time (s)', yticks=[])
ax_rs.set_title(f'Retrosplenial\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_rs.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Striatum']
img = ax_st.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_st.set(xlabel='Time (s)', yticks=[])
ax_st.set_title(f'Striatum\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_st.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Superior colliculus']
img = ax_sc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_sc.set(xlabel='Time (s)', yticks=[])
ax_sc.set_title(f'Superior colliculus\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_sc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Thalamus']
img = ax_th.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_th.set(xlabel='Time (s)', yticks=[])
ax_th.set_title(f'Thalamus\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_th.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Zona incerta']
img = ax_zi.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_zi.set(xlabel='Time (s)', yticks=[])
ax_zi.set_title(f'Zona incerta\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_zi.plot([0, 0], [-1, 1], ls='--', color='k')


plt.tight_layout()
plt.savefig(join(fig_path, 'per_merged_region'), dpi=300)
plt.savefig(join(fig_path, 'per_merged_region.pdf'))

# %%


# Plot per region
peths_df = peths_df.sort_values(['region', 'modulation'], ascending=[True, False])
f, (ax_po, ax_lp, ax_rt, ax_val, ax_ld) = plt.subplots(1, 5, figsize=(6, 3), dpi=dpi)

these_peths = peths_df[peths_df['acronym'] == 'PO']
img = ax_po.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_po.set(xlabel='Time (s)', yticks=[])
ax_po.set_title(f'PO\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_po.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['acronym'] == 'LP']
img = ax_lp.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_lp.set(xlabel='Time (s)', yticks=[])
ax_lp.set_title(f'LP\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_lp.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['acronym'] == 'RT']
img = ax_rt.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_rt.set(xlabel='Time (s)', yticks=[])
ax_rt.set_title(f'RT\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_rt.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['acronym'] == 'VAL']
img = ax_val.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_val.set(xlabel='Time (s)', yticks=[])
ax_val.set_title(f'VAL\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_val.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['acronym'] == 'LD']
img = ax_ld.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_ld.set(xlabel='Time (s)', yticks=[])
ax_ld.set_title(f'LD\n(n={these_peths.shape[0]})', fontweight='bold', fontsize=10)
ax_ld.plot([0, 0], [-1, 1], ls='--', color='k')

