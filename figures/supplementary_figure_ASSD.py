import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os

from statannot import add_stat_annotation
from statannotations.Annotator import Annotator

df = pd.read_csv('results/all_eval/all_metrics_median.csv')


metric = 'assd'

model_names = {metric: 'BiomedParse', f'medsam_{metric}': 'MedSAM (oracle box)', f'sam_{metric}': 'SAM (oracle box)', 
              f'dino_medsam_{metric}': 'MedSAM (Grounding DINO)', f'dino_sam_{metric}': 'SAM (Grounding DINO)'}
df = df.rename(columns=model_names)

score_vars = list(model_names.values())

# filter outlier values
df = df[df['MedSAM (oracle box)'] < 1e10]

modality_list = ['CT', 'MRI', 'X-Ray', 'Pathology', 'Ultrasound', 'Fundus', 'Endoscope', 'Dermoscopy', 'OCT']
# modify modality names
mod_names = {'CT': 'CT', 'MRI': 'MRI', 'MRI-T2': 'MRI', 'MRI-ADC': 'MRI', 'MRI-FLAIR': 'MRI', 'MRI-T1-Gd': 'MRI', 'X-Ray': 'X-Ray', 'pathology': 'Pathology', 
             'ultrasound': 'Ultrasound', 'fundus': 'Fundus', 'endoscope': 'Endoscope', 'dermoscopy': 'Dermoscopy', 'OCT': 'OCT', 'All': 'All'}
df['modality'] = df['modality'].apply(lambda x: mod_names[x])

# add an "All" modality 
all_df = df.copy()
all_df['modality'] = 'All'
df = pd.concat([df, all_df])

df_long = df[['modality', 'task']+score_vars].melt(id_vars=['modality', 'task'], var_name='Model', value_name='Performance')



# add statistical annotations
fig, ax = plt.subplots(figsize=(9, 6))
ax = sns.boxplot(data=df_long, x='modality', y='Performance', hue='Model', ax=ax, palette='Set2', 
            order=['All']+modality_list,
            whis=2, saturation=0.6, linewidth=0.8, fliersize=0.5)  # whiskers at 5th and 95th percentile)
            #errorbar='sd', capsize=0.1, errwidth=1.5)

# no frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# add arrow on y axis
ax.annotate('', xy=(0, 1.05), xytext=(0, -0.01), arrowprops=dict(arrowstyle='->', lw=1, color='black'), xycoords='axes fraction')


plt.title('')
if metric == 'dice':
    plt.ylabel('Dice score', fontsize=18)
elif metric == 'assd':
    plt.ylabel('ASSD', fontsize=18)
plt.xlabel('')
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=14)

# axis thickness
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)


# change to log scale
if metric == 'assd':
    plt.yscale('log')

# set legend names
ax.legend(score_vars, fontsize=14)

# legend on top in a row, without frame
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=2, fontsize=14, frameon=False)

# Define pairs between models for each modality
box_pairs = []

# Add statistical annotations for each modality
for modality in ['All']+modality_list:
    # Define pairs between models within the same modality
    box_pairs += [((modality, 'BiomedParse'), (modality, 'MedSAM (oracle box)'))]
annotator = Annotator(ax, box_pairs, data=df_long, x='modality', y='Performance', hue='Model', 
                      order=['All']+modality_list)
annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
annotator.apply_test(alternative='less')
annotator.annotate()

plt.tight_layout()

# save the plot
ax.get_figure().savefig(f'plots/{metric}_comparison.png')
ax.get_figure().savefig(f'plots/{metric}_comparison.pdf', bbox_inches='tight')