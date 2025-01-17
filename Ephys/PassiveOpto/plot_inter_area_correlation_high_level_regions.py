#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:24:48 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import ttest_1samp
from serotonin_functions import paths, figure_style
import networkx as nx

# Settings
BIN_SIZE = 100
MIN_SUBJECTS = 2
TIME_WIN = [0.2, 0.8]

# Get paths
fig_path, save_path = paths()

# Load in data
corr_df = pd.read_csv(join(save_path, f'region_corr_{BIN_SIZE}.csv'))

# Remove nans
corr_df = corr_df[~np.isnan(corr_df['r'])]

# Select sert-cre animals
sert_df = corr_df[corr_df['sert-cre'] == 1]

# Plot all region pairs
colors, dpi = figure_style()
all_regions = np.unique(np.concatenate((np.unique(corr_df['region_1']), np.unique(corr_df['region_2']))))
summary_df = pd.DataFrame()
for r1, region_1 in enumerate(all_regions[:-1]):
    for r2, region_2 in enumerate(all_regions[r1+1:]):

        # Take a slice out of the dataframe
        slice_df = sert_df.loc[(((sert_df['region_1'] == region_1) & (sert_df['region_2'] == region_2))
                                | ((sert_df['region_1'] == region_2) & (sert_df['region_2'] == region_1)))]

        if len(np.unique(slice_df['subject'])) < MIN_SUBJECTS:
            continue

        # Do statistics
        p_values = np.empty(np.unique(slice_df['time']).shape[0])
        for i, time_bin in enumerate(np.unique(slice_df['time'])):
            _, p_values[i] = ttest_1samp(slice_df.loc[(slice_df['time'] == time_bin), 'r_baseline'], 0)

        # Add to summary df
        for i, subject in enumerate(np.unique(slice_df['subject'])):
            r_mean = np.mean(slice_df.loc[(slice_df['subject'] == subject)
                                          & (slice_df['time'] > TIME_WIN[0])
                                          & (slice_df['time'] < TIME_WIN[1]), 'r_baseline'])
            summary_df = pd.concat((summary_df, pd.DataFrame(index=[summary_df.shape[0]+1], data={
                'r': r_mean, 'subject': subject, 'region_pair': f'{region_1}-{region_2}',
                'region_1': region_1, 'region_2': region_2})))


        # Plot this region pair
        f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
        ax1.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
        sns.lineplot(x='time', y='r_baseline', data=slice_df, legend=None, errorbar='se',
                     color='k')
        n_sert = np.unique(slice_df.loc[slice_df['sert-cre'] == 1, 'subject']).shape[0]
        ax1.set(xlabel='Time (s)', ylabel='Baseline subtracted \n pairwise correlation (r)',
                title=f'{region_1} - {region_2} (n={n_sert})', xticks=[-1, 0, 1, 2, 3],
                ylim=[-0.04, 0.04])
        ax1.scatter(np.unique(slice_df['time'])[p_values < 0.05], np.ones(np.sum(p_values < 0.05)) * 0.038,
                    color='k', marker='*', s=2)

        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, 'Ephys', 'Correlations', 'HighLevelRegions',
                         f'{region_1} - {region_2}.jpg'), dpi=600)
        plt.close(f)

# Get average over animals
mean_summary_df = summary_df.groupby(['region_1', 'region_2']).mean(numeric_only=True).reset_index()

# %% Build network graph
G = nx.Graph()
G.add_nodes_from(np.unique(mean_summary_df['region_1']))
G.add_nodes_from(np.unique(mean_summary_df['region_2']))
for i in mean_summary_df.index:
    if mean_summary_df.loc[i, 'r'] > 0:
        color='r'
    else:
        color='b'
    G.add_edge(mean_summary_df.loc[i, 'region_1'], mean_summary_df.loc[i, 'region_2'],
               weight=mean_summary_df.loc[i, 'r'], color=color)

edge_colors = list(nx.get_edge_attributes(G,'color').values())
weights = np.abs(list(nx.get_edge_attributes(G,'weight').values())) * 400

f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
plt.margins(x=0.2, y=0.2)
pos = nx.circular_layout(G)
nx.draw(G, pos=pos, with_labels=True, width=weights, edge_color=edge_colors, font_size=7,
        node_size=100, node_color='orange', ax=ax1)
plt.savefig(join(fig_path, 'Ephys', 'Correlations', 'HighLevelRegions', 'Correlation_Graph.jpg'),
            dpi=600)

# %% Plot hippocampus frontal / sensory
slice1_df = sert_df.loc[(((sert_df['region_1'] == 'Hippocampus') & (sert_df['region_2'] == 'Frontal'))
                         | ((sert_df['region_1'] == 'Frontal') & (sert_df['region_2'] == 'Hippocampus')))]
slice1_df['region_pair'] = 'Frontal'
slice2_df = sert_df.loc[(((sert_df['region_1'] == 'Hippocampus') & (sert_df['region_2'] == 'Sensory'))
                         | ((sert_df['region_1'] == 'Sensory') & (sert_df['region_2'] == 'Hippocampus')))]
slice2_df['region_pair'] = 'Sensory'
slice_df = pd.concat((slice1_df, slice2_df))

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -1), 1, 2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='r_baseline', data=slice_df, hue='region_pair',
             hue_order=['Frontal', 'Sensory'],
             palette=[colors['OFC'], colors['VIS']], legend='brief', errorbar='se')
ax1.set(xlabel='Time (s)', ylabel='Correlation (r)',
        xticks=[-1, 0, 1, 2, 3], ylim=[-0.02, 0.06])
leg = ax1.legend(title='Hippocampus with:', prop={'size': 5}, frameon=False, bbox_to_anchor=(0.47, 1))
plt.setp(leg.get_title(),fontsize=5)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'Correlations', 'HighLevelRegions', 'Hippocampus_Sensory_Frontal.jpg'),
            dpi=600)



