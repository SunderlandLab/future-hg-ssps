import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import json

print('--------------------------------------------------------------')
print('- SOURCE-RECEPTOR MATRICES - FIG S5 --------------------------')
print('--------------------------------------------------------------')

def prep_SR(SR_path='../output/Source-Receptor_Matrix_Hg0.csv'):
    SR = pd.read_csv(SR_path)

    # columns which are sources and receptors
    SR_cols = ['AfricaMidEast', 'Asia', 'Europe', 'FmrUSSR', 'NorthAm', 'Oceania','SouthAm']
    # columns which are receptors only
    R_cols = ['OCEAN', 'OTHER']
    Natural_cols = ['E_LAND', 'E_OCEAN']

    col_order = ['E_'+c for c in SR_cols]
    if 'Hg0' in SR_path:
        col_order+=Natural_cols

    row_order = ['D_'+c for c in SR_cols]+['D_'+c for c in R_cols]

    SR[col_order] = SR[col_order]/SR[SR['Receptor']=='Total'][col_order].values
    SR = SR.set_index(SR['Receptor'])
    SR = SR.reindex(row_order)
    SR = SR[col_order]
    SR = (SR*100) # convert to percent
    return SR, SR_cols, R_cols, Natural_cols


label_dict = {'AfricaMidEast':'AFM', 'Asia':'ASA', 'Europe':'EUR', 'FmrUSSR':'FSU', 
              'NorthAm':'NAM', 'Oceania':'OCA', 'SouthAm':'SAM',
              'OCEAN':'OCE', 'OTHER':'OTH', 'E_LAND':'Soil', 'E_OCEAN':'Ocean'}

# make ticks smaller
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5

str_fmt = '.1f'

# add frame to heatmap
def add_frame(ax, lw=1, color='k'):
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color(color)
    return

# add frame to colorbar
def add_frame_cbar(ax, lw=1, color='k'):
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color(color)
    return

with open('../data/colormaps/craterlake_shore.json', 'r') as f:
    cmap = json.load(f)
cmap = ListedColormap(cmap['colors'])
cmap_oc = ListedColormap(cmap.colors[::-1])

with open('../data/colormaps/grayC.json', 'r') as f:
    cmap = json.load(f)
cmap = ListedColormap(cmap['colors'])

def make_ocean_matrix(SR, SR_cols, vlim=[0.4,0.6], species='Hg0', add_xlabel=True, add_ylabel=True):
    if species == 'Hg0':
        sns.heatmap(SR.iloc[len(SR_cols):-1, :], ax=ax, vmin=vlim[0], vmax=vlim[1],
                    cmap=cmap_oc, 
                    annot=True, fmt=str_fmt, annot_kws={"size":14},
                    linewidth=0.5, linecolor='k')
    elif species == 'Hg2':
        sns.heatmap(SR.iloc[len(SR_cols):-1], ax=ax, vmin=vlim[0], vmax=vlim[1],
                    cmap=cmap_oc, 
                    annot=True, fmt=str_fmt, annot_kws={"size":14},
                    linewidth=0.5, linecolor='k')     
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # create frame around colorbar
    add_frame_cbar(cbar.ax, lw=0.5, color='k')
    
    if add_ylabel:
        ax.set_ylabel('Receptor Region', fontsize=20, labelpad=20, c='0.1')
    else:
        ax.set_ylabel('')
    if add_xlabel:
        ax.set_xlabel('Source Region', fontsize=20, labelpad=20, c='0.1')
    else:
        ax.set_xlabel('')

    ax.set_yticklabels(['Ocean'], size=12, rotation=0)
    
    if species == 'Hg0':
        labels = SR_cols+['E_LAND', 'E_OCEAN']
        labels = [label_dict[l] for l in labels]
        ax.set_xticklabels(labels, size=12)
    elif species == 'Hg2':
        labels = SR_cols
        labels = [label_dict[l] for l in labels]
        ax.set_xticklabels(labels, size=12)
    return

def make_region_matrix(SR, SR_cols, vlim=[0,60], species='Hg0', add_xlabel=True):
    if species == 'Hg0':
        sns.heatmap(SR.iloc[0:len(SR_cols), 0:len(SR_cols)+2], ax=ax, vmin=vlim[0], vmax=vlim[1],
                    cmap=cmap,
                    annot=True, fmt=str_fmt, annot_kws={"size":14},
                    linewidth=0.5, linecolor='k')
    elif species == 'Hg2':
        sns.heatmap(SR.iloc[0:len(SR_cols), 0:len(SR_cols)], ax=ax, vmin=vlim[0], vmax=vlim[1],
                    cmap=cmap,
                    annot=True, fmt=str_fmt, annot_kws={"size":14},
                    linewidth=0.5, linecolor='k')   

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    add_frame_cbar(cbar.ax, lw=0.5, color='k')
    
    ax.set_ylabel('Receptor Region', fontsize=20, labelpad=20, c='0.1')
    if add_xlabel:
        ax.set_xlabel('Source Region', fontsize=20, labelpad=20, c='0.1')
    else:
        ax.set_xlabel('')
    labels = [label_dict[l] for l in SR_cols]
    ax.set_yticklabels(labels, size=12)

    if species == 'Hg0':
        labels = SR_cols+['E_LAND', 'E_OCEAN']
        labels = [label_dict[l] for l in labels]
        ax.set_xticklabels(labels, size=12)
    elif species == 'Hg2':
        labels = SR_cols
        labels = [label_dict[l] for l in labels]
        ax.set_xticklabels(labels, size=12)
    return

framewidth = 0.5

plt.figure(figsize=(20,9), facecolor='white')
grid = plt.GridSpec(10, 20, wspace=0.1, hspace=0.1)
# plot region Hg0
ax = plt.subplot(grid[:8, :10])
SR, SR_cols, R_cols, Natural_cols = prep_SR(SR_path='../data/output/Source-Receptor_Matrix_Hg0.csv')
make_region_matrix(SR, SR_cols, vlim=[0,20], species='Hg0', add_xlabel=False)
add_frame(ax, lw=framewidth, color='k')
# add title
ax.text(0.5, 1.1, 'Hg$^{0}$ Emission', fontsize=22, transform=ax.transAxes, c='0.1', va='top', ha='center')
# add label
ax.text(-0.08, 1.05, 'a', fontsize=30, weight='bold', transform=ax.transAxes, c='0.0', va='center', ha='left')

cbar = ax.collections[0].colorbar
cbar.set_ticks([0, 5, 10, 15, 20])
cbar.set_ticklabels(['0', '5', '10', '15', '20'])

# plot region Hg2
ax = plt.subplot(grid[:8, 12:])
SR, SR_cols, R_cols, Natural_cols = prep_SR(SR_path='../data/output/Source-Receptor_Matrix_Hg2.csv')
make_region_matrix(SR, SR_cols, vlim=[0,80], species='Hg2', add_xlabel=False)
add_frame(ax, lw=framewidth, color='k')
ax.text(0.5, 1.1, 'Hg$^\mathrm{II}$ Emission', fontsize=22, transform=ax.transAxes, c='0.1', va='top', ha='center')
ax.text(-0.1, 1.05, 'b', fontsize=30, weight='bold', transform=ax.transAxes, c='0.0', va='center', ha='left')

# plot ocean Hg0
ax = plt.subplot(grid[9:, :10])
SR, SR_cols, R_cols, Natural_cols = prep_SR(SR_path='../data/output/Source-Receptor_Matrix_Hg0.csv')
make_ocean_matrix(SR, SR_cols, vlim=[60,80], species='Hg0', add_xlabel=True, add_ylabel=False)
add_frame(ax, lw=framewidth, color='k')
ax.text(-0.08, 1.17, 'c', fontsize=30, weight='bold', transform=ax.transAxes, c='0.0', va='center', ha='left')

ax = plt.subplot(grid[9:, 12:])
SR, SR_cols, R_cols, Natural_cols = prep_SR(SR_path='../data/output/Source-Receptor_Matrix_Hg2.csv')
make_ocean_matrix(SR, SR_cols, vlim=[20,40], species='Hg2', add_xlabel=True, add_ylabel=False)
add_frame(ax, lw=framewidth, color='k')
ax.text(-0.1, 1.17, 'd', fontsize=30, weight='bold', transform=ax.transAxes, c='0.0', va='center', ha='left')

plt.savefig(f'../data/figures/supporting_information/Fig_S6_source_receptor_matrices.pdf')