# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:25:01 2022

@author: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import paths, load_subjects, figure_style

# Settings
asy_tb = 6
DIAGONALS = 3
REGION_PAIRS = ['M2-mPFC', 'M2-ORB', 'mPFC-Amyg', 'ORB-Amyg']

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in data
jpecc_df = pd.read_pickle(join(save_path, 'jPECC.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    jpecc_df.loc[jpecc_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
jpecc_df = jpecc_df[jpecc_df['sert-cre'] == 1]

# Get time axis
time_ax = jpecc_df['time'].mean()
time_asy = time_ax[asy_tb:-asy_tb] + ((asy_tb / 2) * np.mean(np.diff(time_ax)))

# Get 3D array of all jPECC
jPECC, asym, diag_df = dict(), dict(), pd.DataFrame()
for i, rp in enumerate(REGION_PAIRS):
    jPECC[rp] = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == rp, 'r_opto'].to_numpy())

    # Calculate asymmetry
    asym[rp] = np.empty((jPECC[rp].shape[2], jPECC[rp].shape[0] - (asy_tb*2)))
    for jj in range(jPECC[rp].shape[2]):
        for ii, kk in enumerate(range(asy_tb, jPECC[rp].shape[0] - asy_tb)):
            arr_slice = jPECC[rp][kk - asy_tb : kk + asy_tb, kk - asy_tb : kk + asy_tb, jj]
            asym[rp][jj, ii] = (np.median(arr_slice[np.triu_indices(arr_slice.shape[0], k=1)])
                                 - np.median(arr_slice[np.tril_indices(arr_slice.shape[0], k=-1)]))

    # Take diagonal lines (collapse over both time axes and take average)
    # Trigger warning: this is extremely ugly code
    for jj in range(jPECC[rp].shape[2]):
        for ii in np.arange(-DIAGONALS, DIAGONALS+1):
            diag_df = pd.concat((diag_df, pd.DataFrame(data={
                'diag': np.diagonal(jPECC[rp][:,:,jj], offset=ii),
                'diag_bl': (np.diagonal(jPECC[rp][:,:,jj], offset=ii)
                            - np.median(np.diagonal(jPECC[rp][:,:,jj], offset=ii)[time_ax[np.abs(ii):] < 0])),
                'subject': jj, 'region_pair': rp, 'time': time_ax[np.abs(ii):]})))
            if ii == 0:
                this_time = time_ax
            else:
                this_time = time_ax[:-np.abs(ii)]
            this_times = time_ax[:-np.abs(ii)]
            diag_df = pd.concat((diag_df, pd.DataFrame(data={
                'diag': np.diagonal(jPECC[rp][:,:,jj], offset=ii),
                'diag_bl': (np.diagonal(jPECC[rp][:,:,jj], offset=ii)
                            - np.median(np.diagonal(jPECC[rp][:,:,jj], offset=ii)[this_time < 0])),
                'subject': jj, 'region_pair': rp, 'time': this_time})))

# Take average per timepoint
diag_df = diag_df.groupby(['time', 'subject', 'region_pair']).mean()
diag_df = diag_df.melt(ignore_index=False).reset_index()


# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2, ax_cb) = plt.subplots(1, 3, figsize=(3.5, 1.75),
                                    gridspec_kw={'width_ratios': [1, 1, 0.2]}, dpi=dpi)

ax1.imshow(np.flipud(np.mean(jPECC['M2-mPFC'], axis=2)), vmin=-0.5, vmax=0.5, cmap='icefire',
           extent=[time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2,
                   time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2])
ax1.invert_xaxis()
ax1.plot([0, 0], [-1, 3], color='white')
#ax1.plot([1, 1], [-1, 2], color='white')
ax1.plot([-1, 3], [0, 0], color='white')
#ax1.plot([-1, 2], [1, 1], color='white')
ax1.plot([-1, 3], [-1, 3], color='white')
ax1.set(ylabel='M2', xlabel='mPFC, time from stim. (s)', title='jPECC: mPFC vs M2',
        xlim=[-1, 3], ylim=[-1, 3])

ax2.imshow(np.flipud(np.mean(jPECC['mPFC-Amyg'], axis=2).T), vmin=-0.5, vmax=0.5, cmap='icefire',
           extent=[time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2,
                   time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2])
ax2.invert_xaxis()
ax2.plot([0, 0], [-1, 3], color='white')
#ax2.plot([1, 1], [-1, 2], color='white')
ax2.plot([-1, 3], [0, 0], color='white')
#ax2.plot([-1, 2], [1, 1], color='white')
ax2.plot([-1, 3], [-1, 3], color='white')
ax2.set(ylabel='Amygdala', xlabel='mPFC, time from stim. (s)', title='mPFC vs Amygdala',
        xlim=[-1, 3], ylim=[-1, 3])

ax_cb.axis('off')
plt.tight_layout()
cb_ax = f.add_axes([0.8, 0.3, 0.01, 0.5])
cbar = f.colorbar(mappable=ax2.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Population correlation (r)', rotation=270, labelpad=10)

plt.savefig(join(fig_path, 'jPECC_mPFC_M2_Amyg.pdf'))

# %%

colors, dpi = figure_style()
f, (ax1, ax2, ax_cb) = plt.subplots(1, 3, figsize=(3.5, 1.75),
                                    gridspec_kw={'width_ratios': [1, 1, 0.2]}, dpi=dpi)

ax1.imshow(np.flipud(np.mean(jPECC['M2-ORB'], axis=2)), vmin=-0.5, vmax=0.5, cmap='icefire',
           extent=[time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2,
                   time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2])
ax1.invert_xaxis()
ax1.plot([0, 0], [-1, 2], color='white')
#ax1.plot([1, 1], [-1, 2], color='white')
ax1.plot([-1, 2], [0, 0], color='white')
#ax1.plot([-1, 2], [1, 1], color='white')
ax1.plot([-1, 2], [-1, 2], color='white')
ax1.set(ylabel='M2', xlabel='ORB, time from stim. (s)', title='jPECC: ORB vs M2',
        xlim=[-1, 2], ylim=[-1, 2])

ax2.imshow(np.flipud(np.mean(jPECC['ORB-Amyg'], axis=2).T), vmin=-0.5, vmax=0.5, cmap='icefire',
           extent=[time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2,
                   time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2])
ax2.invert_xaxis()
ax2.plot([0, 0], [-1, 2], color='white')
#ax2.plot([1, 1], [-1, 2], color='white')
ax2.plot([-1, 2], [0, 0], color='white')
#ax2.plot([-1, 2], [1, 1], color='white')
ax2.plot([-1, 2], [-1, 2], color='white')
ax2.set(ylabel='Amygdala', xlabel='ORB, time from stim. (s)', title='ORB vs Amygdala',
        xlim=[-1, 2], ylim=[-1, 2])

ax_cb.axis('off')
plt.tight_layout()
cb_ax = f.add_axes([0.8, 0.3, 0.01, 0.5])
cbar = f.colorbar(mappable=ax2.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Population correlation (r)', rotation=270, labelpad=10)

plt.savefig(join(fig_path, 'jPECC_ORB_M2_Amyg.pdf'))

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -1), 1, 3, color='royalblue', alpha=0.25, lw=0))
ax1.plot([-1, 2], [0, 0], ls='--', color='grey')
ax1.plot(time_asy, np.mean(asym['M2-mPFC'], axis=0), color=colors['M2'], label='M2')
ax1.fill_between(time_asy,
                 np.mean(asym['M2-mPFC'], axis=0)-(np.std(asym['M2-mPFC'], axis=0)/np.sqrt(asym['M2-mPFC'].shape[0])),
                 np.mean(asym['M2-mPFC'], axis=0)+(np.std(asym['M2-mPFC'], axis=0)/np.sqrt(asym['M2-mPFC'].shape[0])),
                 alpha=0.2, color=colors['M2'])
ax1.plot(time_asy, np.mean(asym['mPFC-Amyg'], axis=0), color=colors['Amyg'], label='Amygdala')
ax1.fill_between(time_asy,
                 np.mean(asym['mPFC-Amyg'], axis=0)-(np.std(asym['mPFC-Amyg'], axis=0)/np.sqrt(asym['mPFC-Amyg'].shape[0])),
                 np.mean(asym['mPFC-Amyg'], axis=0)+(np.std(asym['mPFC-Amyg'], axis=0)/np.sqrt(asym['mPFC-Amyg'].shape[0])),
                 alpha=0.2, color=colors['Amyg'])
leg = ax1.legend(prop={'size': 5}, frameon=True, loc='lower left')
leg.get_frame().set_linewidth(0)
ax1.set(ylabel='jPECC asymmetry', xlabel='Time from stim. onset (s)', title='Medial prefrontal cortex',
        ylim=[-0.3, 0.3], xlim=[-1, 2], yticks=np.arange(-0.3, 0.31, 0.15))

ax2.add_patch(Rectangle((0, -1), 1, 3, color='royalblue', alpha=0.25, lw=0))
ax2.plot([-1, 2], [0, 0], ls='--', color='grey')
ax2.plot(time_asy, np.mean(asym['M2-ORB'], axis=0), color=colors['M2'], label='M2')
ax2.fill_between(time_asy,
                 np.mean(asym['M2-ORB'], axis=0)-(np.std(asym['M2-ORB'], axis=0)/np.sqrt(asym['M2-ORB'].shape[0])),
                 np.mean(asym['M2-ORB'], axis=0)+(np.std(asym['M2-ORB'], axis=0)/np.sqrt(asym['M2-ORB'].shape[0])),
                 alpha=0.2, color=colors['M2'])
ax2.plot(time_asy, np.mean(asym['ORB-Amyg'], axis=0), color=colors['Amyg'], label='Amygdala')
ax2.fill_between(time_asy,
                 np.mean(asym['ORB-Amyg'], axis=0)-(np.std(asym['ORB-Amyg'], axis=0)/np.sqrt(asym['ORB-Amyg'].shape[0])),
                 np.mean(asym['ORB-Amyg'], axis=0)+(np.std(asym['ORB-Amyg'], axis=0)/np.sqrt(asym['ORB-Amyg'].shape[0])),
                 alpha=0.2, color=colors['Amyg'])
leg = ax2.legend(prop={'size': 5}, frameon=True, loc='lower left')
leg.get_frame().set_linewidth(0)
ax2.set(ylabel='jPECC asymmetry', xlabel='Time from stim. onset (s)', title='Orbitofrontal cortex',
        ylim=[-0.3, 0.3], xlim=[-1, 2], yticks=np.arange(-0.3, 0.31, 0.15))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'jPECC_asymmetry_Amyg.pdf'))

# %%
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.plot([-1, 3], [0, 0], ls='--', color='grey')
ax1.add_patch(Rectangle((0, -0.4), 1, 0.8, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=diag_df[diag_df['variable'] == 'diag_bl'], hue='region_pair', ax=ax1, ci=68,
             hue_order=['M2-mPFC', 'mPFC-Amyg'], palette=[colors['M2'], colors['Amyg']])
ax1.set(xlabel='Time (s)', ylabel='Canonical correlation \n over baseline (r)', xlim=[-1, 3], ylim=[-0.4, 0.4],
        yticks=np.arange(-0.4, 0.41, 0.1), title='Medial prefrontal cortex', xticks=[-1, 0, 1, 2, 3])
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['M2', 'Amygdala']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left')
leg.get_frame().set_linewidth(0)

ax2.plot([-1, 3], [0, 0], ls='--', color='grey')
ax2.add_patch(Rectangle((0, -0.4), 1, 0.8, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=diag_df[diag_df['variable'] == 'diag_bl'], hue='region_pair', ax=ax2, ci=68,
             hue_order=['M2-ORB', 'ORB-Amyg'], palette=[colors['M2'], colors['Amyg']])
ax2.set(xlabel='Time (s)', ylabel='Canonical correlation \n over baseline (r)', xlim=[-1, 3], ylim=[-0.4, 0.4],
        yticks=np.arange(-0.4, 0.41, 0.1), title='Orbitofrontal cortex', xticks=[-1, 0, 1, 2, 3])
leg_handles, _ = ax2.get_legend_handles_labels()
leg_labels = ['M2', 'Amygdala']
leg = ax2.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left')
leg.get_frame().set_linewidth(0)

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_CCA_Front_Amyg.pdf'))


