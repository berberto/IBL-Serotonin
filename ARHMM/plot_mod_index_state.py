#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:24:39 2022
By: Guido Meijer
"""
import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import paths, load_subjects, figure_style

fig_path, data_path = paths()

# Load data
state_mod_neurons = pd.read_csv(join(data_path, 'neural_state_modulation.csv'))
opto_mod_neurons = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
all_neurons = pd.merge(state_mod_neurons, opto_mod_neurons, on=['eid', 'pid', 'subject', 'date',
                                                                'neuron_id', 'region'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]

#all_neurons = all_neurons[all_neurons['mod_index_late'] < 0]

all_neurons['abs_mod_inactive'] = all_neurons['mod_index_inactive'].abs()
all_neurons['abs_mod_active'] = all_neurons['mod_index_active'].abs()

per_animal_df = all_neurons[all_neurons['modulated'] == True].groupby('subject').median()

# %%
colors, dpi = figure_style()
sert_colors = [colors['wt'], colors['sert']]
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
for i in per_animal_df.index:
    ax1.plot([0, 1], [per_animal_df.loc[i, 'abs_mod_inactive'], per_animal_df.loc[i, 'abs_mod_active']],
             '-o', color=sert_colors[per_animal_df.loc[i, 'sert-cre'].astype(int)], markersize=2)
ax1.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Inactive', 'Active'], xlabel='State')
sns.despine(trim=True)
plt.tight_layout()