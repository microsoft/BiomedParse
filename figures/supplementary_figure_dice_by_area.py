#%%
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# Define file paths
base_dir = 'results'
eval_results_path = os.path.join(base_dir, 'all_eval/biomedparse_eval_results.json')

# Load data
with open(eval_results_path, 'r') as f:
    parsed_data = json.load(f)

# Extract relevant information
def extract_data(parsed_data):
    records = []
    for dataset in parsed_data:
        dataset_name = dataset[len('biomed_'):-len('_test/grounding_refcoco')]
        instances = parsed_data[dataset]["grounding"]["instance_results"]
        for instance in instances:
            metadata = instance["metadata"]
            grounding_info = metadata["grounding_info"][0]
            record = {
                "dataset": dataset_name,
                "file_name": grounding_info["mask_file"].split("/")[-1],
                "area": grounding_info["area"],
                "bp_dice": instance["Dice"][0]
            }
            records.append(record)
    return pd.DataFrame(records)

df = extract_data(parsed_data)

# Merge with SAM and MedSAM data
def merge_with_sam_medsam(df, parsed_data, base_dir):
    comparison_df = pd.DataFrame()
    for dataset in parsed_data:
        dataset_name = dataset[len('biomed_'):-len('_test/grounding_refcoco')]
        if any(sub in dataset_name for sub in ['MSD', 'Radiography', 'amos22']):
            dataset_name = dataset_name.replace('-', '/')
        
        sam_data_path = os.path.join(base_dir, dataset_name, 'test_sam_vit_b_01ec64_dice.csv')
        medsam_data_path = os.path.join(base_dir, dataset_name, 'test_medsam_dice.csv')
        
        sam_data = pd.read_csv(sam_data_path, delimiter=',')
        medsam_data = pd.read_csv(medsam_data_path, delimiter=',')
        
        merged_data = pd.merge(sam_data, medsam_data, on='image', suffixes=('_sam', '_medsam'))
        merged_data.rename(columns={'image': 'file_name'}, inplace=True)
        merged_data['dataset'] = dataset_name.replace('/', '-')
        
        comparison_df = pd.concat([comparison_df, merged_data], ignore_index=True)
    
    return pd.merge(df, comparison_df, on=['dataset', 'file_name'], how='inner')

df = merge_with_sam_medsam(df, parsed_data, os.path.join(base_dir, 'dataset_results'))

# Save to CSV
df.to_csv(os.path.join(base_dir, 'all_eval/dice_by_size.csv'), index=False)

# Filter datasets
rad_list = [
    'ACDC', 'COVID-QU-Ex', 'CXR_Masks_and_Labels', 'LGG', 'LIDC-IDRI', 'MMs', 
    'MSD-Task01_BrainTumour', 'MSD-Task02_Heart', 'MSD-Task03_Liver', 'MSD-Task04_Hippocampus',
    'MSD-Task05_Prostate', 'MSD-Task06_Lung', 'MSD-Task07_Pancreas', 'MSD-Task08_HepaticVessel',
    'MSD-Task09_Spleen', 'MSD-Task10_Colon', 'PROMISE12', 'QaTa-COV19', 'Radiography-COVID', 
    'Radiography-Lung_Opacity', 'Radiography-Normal', 'Radiography-Viral_Pneumonia', 
    'amos22-CT', 'amos22-MRI', 'kits23', 'COVID-19_CT'
]
df = df[df['dataset'].isin(rad_list)]

# Plot area to Dice ratio
def plot_area_to_dice(df):
    sns.set_theme(style='ticks')

    total_image_area = 1024 * 1024  # pixels
    max_area_threshold = total_image_area  # Adjust this threshold as needed
    filtered_df = df[df['area'] <= max_area_threshold]

    filtered_df['area_percentage'] = (filtered_df['area'] / total_image_area) * 100

    bins = np.linspace(filtered_df['area_percentage'].min(), filtered_df['area_percentage'].max(), 15)
    filtered_df['area_bin'] = pd.cut(filtered_df['area_percentage'], bins)

    avg_dice_bp = filtered_df.groupby('area_bin')['bp_dice'].mean()
    avg_dice_sam = filtered_df.groupby('area_bin')['dice_sam'].mean() if 'dice_sam' in filtered_df.columns else None
    avg_dice_medsam = filtered_df.groupby('area_bin')['dice_medsam'].mean() if 'dice_medsam' in filtered_df.columns else None

    sem_dice_bp = filtered_df.groupby('area_bin')['bp_dice'].apply(sem)
    sem_dice_sam = filtered_df.groupby('area_bin')['dice_sam'].apply(sem) if 'dice_sam' in filtered_df.columns else None
    sem_dice_medsam = filtered_df.groupby('area_bin')['dice_medsam'].apply(sem) if 'dice_medsam' in filtered_df.columns else None

    colors = sns.color_palette("colorblind", 3)

    plt.figure(figsize=(14, 10))

    plt.errorbar(avg_dice_bp.index.categories.mid, avg_dice_bp, yerr=sem_dice_bp, fmt='-o', label='BiomedParse', color=colors[0], capsize=5)
    if avg_dice_sam is not None:
        plt.errorbar(avg_dice_sam.index.categories.mid, avg_dice_sam, yerr=sem_dice_sam, fmt='-o', label='SAM', color=colors[1], capsize=5)
    if avg_dice_medsam is not None:
        plt.errorbar(avg_dice_medsam.index.categories.mid, avg_dice_medsam, yerr=sem_dice_medsam, fmt='-o', label='MedSAM', color=colors[2], capsize=5)

    plt.xlabel('Area (% of total image)', fontsize=20)
    plt.ylabel('Dice Score', fontsize=20)
    plt.grid(False)
    plt.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(filtered_df['area_percentage'].min(), filtered_df['area_percentage'].max())
    sns.despine()

    plt.tight_layout()
    plt.savefig(os.path.join('plots/area_vs_dice.pdf'), dpi=300)
    plt.show()

plot_area_to_dice(df)

# %%
