#%%
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
sectors = {k: 0 for k in hierarchy_data.keys()}
for sector_name in hierarchy_data:
    for k,v in hierarchy_data[sector_name]['child'].items():
        sectors[sector_name] += len(v['child'])
    sectors[sector_name] += 1

name2color = {"organ": "#E41A1C", "abnormality": "#377EB8", "histology": "#4DAF4A"}

def generate_shades(base_color, n):
    return sns.light_palette(base_color, n + 2)[1:-1]

color_schemes = {}
for sector in sectors:
    child_colors = generate_shades(name2color[sector], len(hierarchy_data[sector]['child']))
    color_schemes[sector] = child_colors

parent_track_ratio = (72, 85)
middle_track_ratio =  (85, 100)
bar_track_ratio = (45, 70)
parent_track_font_size = 7
middle_track_font_size = 5.5
bar_track_font_size = 7
outer_track_font_size = 9

circos = Circos(sectors, space=8.8)
for sector in circos.sectors:
    idx2label = {}
    idx = 1
    for k,v in hierarchy_data[sector.name.lower()]['child'].items():
        for k1,v1 in v['child'].items():
            idx2label[idx] = k1
            idx += 1
    idx2label[idx] = ''
    idx2label[0] = ''

    track_outer = sector.add_track((100, 101))
    track_outer.xticks_by_interval(
        1,
        tick_length=0,
        outer=True,
        show_bottom_line=False,
        label_orientation="vertical",
        label_formatter=lambda v: idx2label[int(v)],
        label_size=outer_track_font_size,
        show_endlabel=True
    )

    track = sector.add_track(parent_track_ratio)
    track.axis(fc=name2color[sector.name], lw=0)
    track.text(sector.name.capitalize().replace('Mri', 'MRI').replace('Ct', 'CT').replace('Oct', 'OCT').replace('Dermoscopy', "DS"), color="white", size=parent_track_font_size)

    track1 = sector.add_track(middle_track_ratio, r_pad_ratio=0.1)
    sect_start = 0
    color_idx = 0
    for i, (k,v) in enumerate(hierarchy_data[sector.name.lower()]['child'].items()):
        sect_size = len(v['child']) if i != len(hierarchy_data[sector.name.lower()]['child'])-1 else len(v['child'])+1
        if i == 0:
            sect_size += 0.5
        if i == len(hierarchy_data[sector.name.lower()]['child'])-1:
            sect_size -= 0.5
        track1.rect(sect_start, sect_start+sect_size, r_lim=(middle_track_ratio[0], middle_track_ratio[1]-1), ec="black", lw=0,fc=color_schemes[sector.name][color_idx])
        color_idx += 1
        track1.text(k.replace('abnormality', 'abn.').replace(' anatomies', '').replace(' disturbance', '').replace('other abn.', 'Other').replace('liver', '').replace('pancreas', '').capitalize(), sect_start+sect_size/2, color="black", size=middle_track_font_size)
        sect_start += sect_size

    x = np.linspace(sector.start+1 , sector.end-1 , int(sector.size)-1)
    y = [target_counts[idx2label[i+1]] for i in range(0,len(x))]
    y_box = boxcox(y, 0.35)

    track2 = sector.add_track(bar_track_ratio, r_pad_ratio=0.1)
    track2.axis()
    track2.yticks([1.14, 2.29, 3.43, 4.58], ["10$^2$", "10$^3$", "10$^4$", "10$^5$"], label_size=bar_track_font_size-1)
    track2.bar(x, y_box, color=name2color[sector.name], alpha=0.5, align="center", lw=0)

fig = circos.plotfig()
fig.savefig('plots/figure_1a.pdf')
plt.show()

# %%
