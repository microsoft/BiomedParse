#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json, os
import seaborn as sns

plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})

# Load data
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
base_dir = 'metadata'
data = load_data(os.path.join(base_dir, 'modality_counts.json'))
separate_submodality = False

# Transform data for plotting
def transform_data(data):
    df = pd.DataFrame([(modality, subcat, count) for modality, subcats in data.items() for subcat, count in subcats.items()], columns=['Modality', 'Sub-category', 'Count'])
    return df

df = transform_data(data)

# Calculate total counts by modality and sort
def calculate_totals(df):
    total_counts_by_modality = df.groupby("Modality")["Count"].sum().sort_values(ascending=True)
    sorted_modalities = total_counts_by_modality.index.tolist()
    return total_counts_by_modality, sorted_modalities

total_counts_by_modality, sorted_modalities = calculate_totals(df)

# Generate color map
def generate_color_map(total_counts_by_modality):
    base_colors = plt.cm.cool(np.linspace(0, 1, len(total_counts_by_modality)))
    modality_color_map = {modality: base_colors[i] for i, modality in enumerate(total_counts_by_modality.index)}
    return modality_color_map

modality_color_map = generate_color_map(total_counts_by_modality)

# Format total count for display
def format_total_count(total_count):
    if total_count >= 1000:
        exponent = int(np.floor(np.log10(total_count)))
        mantissa = total_count / 10**exponent
        formatted_total = f'{mantissa:.2f} x 10$^{exponent}$'
    else:
        exponent = 0
        formatted_total = str(total_count)
    return formatted_total, exponent

# Plotting function
def plot_data(df, total_counts_by_modality, sorted_modalities, modality_color_map, separate_submodality):
    fig, ax = plt.subplots(figsize=(10, 12))
    current_bottom = np.zeros(len(sorted_modalities))
    gap = 0.005 if separate_submodality else 0
    shades = np.power(np.linspace(0.75, 1, df.groupby("Sub-category").ngroups), 2)

    if separate_submodality:
        for i, modality in enumerate(sorted_modalities):
            subdf = df[df["Modality"] == modality].sort_values(by='Count', ascending=False)
            for j, (index, row) in enumerate(subdf.iterrows()):
                count = row['Count']
                if count > 0:
                    color = np.array(modality_color_map[modality]) * shades[j % len(shades)]
                    ax.barh(modality, count, left=current_bottom[i], color=color, height=0.8, log=True, edgecolor='white', linewidth=0.5)
                    current_bottom[i] += count + gap
            current_bottom[i] -= gap
            total_count = total_counts_by_modality[modality]
            formatted_total, exponent = format_total_count(total_count)
            ax.text(current_bottom[i] + (10**exponent)*0.05, i, formatted_total, va='center', fontsize=20, ha='left')
    else:
        for i, modality in enumerate(sorted_modalities):
            total_count = total_counts_by_modality[modality]
            color = np.array(modality_color_map[modality] * shades[0])
            if modality.islower():
                modality = modality.capitalize()
            ax.barh(modality, total_count, color=color, height=0.8, log=True, edgecolor='white', linewidth=0.5)
            formatted_total, exponent = format_total_count(total_count)
            ax.text(total_count + (10**exponent)*0.05, i, formatted_total, va='center', fontsize=20, ha='left')

    configure_plot(ax, sorted_modalities)

    plt.tight_layout()
    plt.savefig("plots/data_dist_modality_bar_subbar.pdf" if separate_submodality else "plots/data_dist_modality_bar.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

# Configure plot aesthetics
def configure_plot(ax, sorted_modalities):
    ax.set_xscale('log')
    ax.set_title("Number of images per modality", fontsize=28)
    plt.yticks(rotation=0, fontsize=24, va='center')
    ax.tick_params(axis='x', which='major', length=8)
    ax.tick_params(axis='x', which='minor', length=5)
    plt.xticks(fontsize=24)
    sns.despine()

# Main script execution
plot_data(df, total_counts_by_modality, sorted_modalities, modality_color_map, separate_submodality)

# %%
