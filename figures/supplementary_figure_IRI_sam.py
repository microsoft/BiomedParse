import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os

from statannot import add_stat_annotation
from statannotations.Annotator import Annotator

df = pd.read_csv('results/all_eval/all_metrics_mean.csv')

# modify modality names
mod_names = {'CT': 'CT', 'MRI': 'MRI', 'MRI-T2': 'MRI', 'MRI-ADC': 'MRI', 'MRI-FLAIR': 'MRI', 'MRI-T1-Gd': 'MRI', 'X-Ray': 'X-Ray', 'pathology': 'Pathology', 
             'ultrasound': 'Ultrasound', 'fundus': 'Fundus', 'endoscope': 'Endoscope', 'dermoscopy': 'Dermoscopy', 'OCT': 'OCT', 'All': 'All'}
df['modality'] = df['modality'].apply(lambda x: mod_names[x])

# MedSAM reported tasks
reported_baseline_df = pd.read_csv('results/all_eval/reported_baseline_tasks.csv')

# find overlap between the dfs by dataset and target
overlap_df = pd.merge(df, reported_baseline_df, on=['task', 'modality', 'site', 'target'], 
                      suffixes=('_biomedparse', '_baseline'))
# non-overlapping datasets
non_overlap_df = df[~df['task'].isin(overlap_df['task'])]



baseline = 'sam'
metric = 'IRI'

baseline_names = {'medsam': 'MedSAM', 'sam': 'SAM'}
metric_names = {'box_ratio': 'Box Ratio', 'convex_ratio': 'Convex Ratio', 
                'IRI': 'Inversed Rotational Inertia'}

non_overlap_df['diff'] = non_overlap_df[f'dice'] - non_overlap_df[f'{baseline}_dice']
# scatter plot
fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(data=non_overlap_df, x=metric, y='diff', ax=ax, markers='o', s=80)

# add linear regression line
sns.regplot(data=non_overlap_df, x=metric, y='diff', ax=ax, scatter=False, 
            color='k', line_kws={'linestyle':'--', 'linewidth':1})

# remove all spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)


# add arrow on x-axis and y-axis
xlim = [0, 1.05]
ylim = [-0.06, 0.79]
ax.annotate('', xy=(xlim[1], ylim[0]), xytext=(xlim[0], ylim[0]), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(xlim[0], ylim[1]), xytext=(xlim[0], ylim[0]), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.xaxis.set_tick_params(width=1.5)
ax.yaxis.set_tick_params(width=1.5)

# set x-ticks and y-ticks
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# show R^2 value, p value, and equation of the line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(non_overlap_df[metric], non_overlap_df['diff'])
x_text = 0.4
plt.text(x_text, 0.84, f'$R^2={r_value**2:.2f}$', fontsize=20, transform=ax.transAxes)
plt.text(x_text, 0.77, f'$p={p_value:.2e}$', fontsize=20, transform=ax.transAxes)
plt.text(x_text, 0.7, f'$y={slope:.2f}x+{intercept:.2f}$', fontsize=20, transform=ax.transAxes)

plt.title('')
plt.ylabel(f'Improvement over {baseline_names[baseline]}', fontsize=20)
plt.xlabel(f'{metric_names[metric]}', fontsize=22)

plt.tight_layout()

# save the plot
ax.get_figure().savefig(f'plots/{metric}_mean_improvement_{baseline}.png')
ax.get_figure().savefig(f'plots/{metric}_mean_improvement_{baseline}.pdf', bbox_inches='tight')

