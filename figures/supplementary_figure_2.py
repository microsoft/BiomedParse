# %%
import os
import json
import numpy as np
import seaborn as sns
from scipy.stats import boxcox
from pycirclize import Circos
import matplotlib.pyplot as plt

base_dir = 'metadata'
with open(os.path.join(base_dir,'hierarchy.json'), 'r') as f:
    hierarchy_data = json.load(f)

with open(os.path.join(base_dir,'target_counts.json'), 'r') as f:
    target_counts = json.load(f)

with open(os.path.join(base_dir,'modality_counts.json'), 'r') as f:
    modality_counts = json.load(f)

# color scheme
sectors = {k: len(v) for k,v in modality_counts.items()}
name2color = {
    "MRI": "#005A9E",
    "CT": "#FF7F00",
    "pathology": "#984EA3",
    "ultrasound": "#7BC8F6",
    "X-Ray": "#999999",
    "fundus": "#76B041",
    "dermoscopy": "#FDBF6F",
    "endoscope": "#C0392B",
    "OCT": "#33A02C",
}

def generate_shades(base_color, n):
    return sns.light_palette(base_color, n + 2)[1:-1]

color_schemes = {}
for sector in sectors:
    child_colors = generate_shades(name2color[sector], len(modality_counts[sector]))
    color_schemes[sector] = child_colors

parent_track_ratio = (72, 85)
middle_track_ratio =  (85, 100)
bar_track_ratio = (45, 70)
parent_track_font_size = 7
middle_track_font_size = 5.5
bar_track_font_size = 7

circos = Circos(sectors, space=6)
for sector in circos.sectors:
    track = sector.add_track(parent_track_ratio)
    track.axis(fc=name2color[sector.name], lw=0)
    track.text(sector.name.capitalize().replace('Mri', 'MRI').replace('Ct', 'CT').replace('Oct', 'OCT').replace('Dermoscopy', "DS"), color="white", size=parent_track_font_size)

    track1 = sector.add_track(middle_track_ratio, r_pad_ratio=0.1)
    sect_start = 0
    color_idx = 0
    for k,v in modality_counts[sector.name].items():
        sect_size = 1
        track1.rect(sect_start, sect_start+sect_size, r_lim=(middle_track_ratio[0], middle_track_ratio[1]-1) , ec="black", lw=0,fc=color_schemes[sector.name][color_idx])
        color_idx += 1
        track1.text(k.capitalize(), sect_start+sect_size/2, color="black", size=middle_track_font_size)
        sect_start += sect_size

    x = np.linspace(sector.start+0.5, sector.end-0.5, int(sector.size))
    y = [v for k,v in modality_counts[sector.name].items()]
    y_box = boxcox(y, 0.35)

    track2 = sector.add_track(bar_track_ratio, r_pad_ratio=0.1)
    track2.axis()
    track2.yticks([1.14, 2.29, 3.43, 4.58], ["10$^2$", "10$^3$", "10$^4$", "10$^5$"], label_size=bar_track_font_size-1)
    track2.bar(x, y_box, color=name2color[sector.name], alpha=0.5, align="center", lw=0)

fig = circos.plotfig()
fig.savefig('plots/data_target_modality.pdf')
plt.show()

# %%
