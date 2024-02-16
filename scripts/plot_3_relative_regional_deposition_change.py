import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

print('--------------------------------------------------------------')
print('- RELATIVE REGIONAL DEPOSITION CHANGE - FIG 3 ----------------')
print('--------------------------------------------------------------')

region_hex_codes = ['#aa3f2d', '#e0962f', '#f2cf45', '#c7deb2', '#78b24f', '#4c8dd4', '#0f4064']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=region_hex_codes)

df = pd.read_csv('../data/output/regional_deposition_attribution.csv')

region_areas = pd.read_csv('../data/region_masks/region_mask_areas.csv')

def get_region_area(region:str, region_areas=region_areas):
    return region_areas[region_areas['region']==region]['area [m2]'].values.item()

dep = df[['scenario','year','receptor_region','receptor_deposition_total','units']].drop_duplicates()

# subset to anthropogenic receptors -- i.e. exclude OCEAN and OTHER
# do this because `region_mask_areas.csv` only has areas for anthropogenic regions
anthro_receptors = ['AfricaMidEast','Asia','Europe','FmrUSSR','NorthAm','Oceania','SouthAm']
dep = dep[dep['receptor_region'].isin(anthro_receptors)]

dep.rename(columns={'receptor_deposition_total':'receptor_deposition_total_Mg'}, inplace=True)

# define lambda function to add area as column based on value in `receptor_region`
get_area = lambda x: get_region_area(x)
dep['receptor_region_area_m2'] = dep['receptor_region'].apply(lambda x: get_area(x))

# calculate deposition per unit area
dep['receptor_region_ug_m2_yr'] = (dep['receptor_deposition_total_Mg']*1e12)/dep['receptor_region_area_m2']

# calculate total deposition and total area for all anthropogenic regions
anthro_total = dep.groupby(by=['scenario','year'], as_index=False).sum(numeric_only=True)[['scenario', 'year', 'receptor_deposition_total_Mg','receptor_region_area_m2', ]]
anthro_total['receptor_region_ug_m2_yr'] = anthro_total['receptor_deposition_total_Mg']/anthro_total['receptor_region_area_m2']*1e12
anthro_total['receptor_region'] = 'all_anthro'

# append all_anthro rows `dep`
dep = pd.concat((dep, anthro_total))
dep = dep.sort_values(['scenario','year','receptor_region'])
dep = dep.reset_index(drop=True)

dep_out = pd.DataFrame()
for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
    for region in dep['receptor_region'].unique():
        sel = dep[(dep['scenario']==scenario) & (dep['receptor_region']==region)].copy()
        baseline = sel.loc[sel['year']==2010, 'receptor_deposition_total_Mg'].values[0]
        sel['change_relative_to_2010'] = (sel['receptor_deposition_total_Mg'] - baseline)/baseline
        sel['change_relative_to_2010 [%]'] = sel['change_relative_to_2010']*100
        dep_out = pd.concat((dep_out, sel))

dep_out.rename(columns={'receptor_deposition_total_Mg':'deposition [Mg/yr]',
                        'receptor_region': 'receptor region',
                        'receptor_region_area_m2':'area [m2]',
                        'receptor_region_ug_m2_yr': 'deposition [ug/m2/yr]',
                        'change_relative_to_2010 [%]': 'change since 2010 [%]'}, inplace=True)
dep_out = dep_out[['scenario','year','receptor region','deposition [Mg/yr]','area [m2]','deposition [ug/m2/yr]','change since 2010 [%]']]
for column in ['deposition [Mg/yr]','deposition [ug/m2/yr]','change since 2010 [%]']:
    dep_out[column] = dep_out[column].round(2)

dep_out = dep_out[['scenario', 'year', 'receptor region', 'deposition [Mg/yr]', 'deposition [ug/m2/yr]', 'change since 2010 [%]', 'area [m2]']]
dep_out.sort_values(['scenario','year','receptor region'], inplace=True)
dep_out = dep_out.reset_index(drop=True)
dep_out.to_csv('../data/output/regional_deposition_trends.csv', index=False)

def make_panel(ax, dep:pd.DataFrame, scenario:str):
    labelsize = 8 # per AGU style guide

    sel = dep[(dep['scenario']==scenario)]
    offsets = np.linspace(-0.5, 6.5, len(anthro_receptors))

    for (receptor_region, offset) in zip(anthro_receptors, offsets):
        tmp = sel[sel['receptor_region']==receptor_region]
        d0 = tmp[tmp['year']==2010]['receptor_deposition_total_Mg'].values[0]
        d  = tmp['receptor_deposition_total_Mg']
        delta = d - d0
        relative_delta = delta/d0

        ax.bar(x=tmp['year']+offset, height=relative_delta, edgecolor='k', lw=0.3, label=receptor_region, zorder=2)

    for x in np.arange(2020, 2110, 10):
        ax.axvspan(x-1.5, x+7.5, color='k', linewidth=0, alpha=0.02)

    ax.axhline(0, c='k', lw=0.8)
    ax.set_xlim(2017, 2108.5)
    offset_mean = np.mean(offsets)
    ax.set_xticks(ticks=np.arange(2020+offset_mean, 2110+offset_mean, 10));
    ax.set_xticklabels(labels=np.arange(2020, 2110, 10), fontsize=labelsize);
    ax.set_ylim(-0.76, 0.35)
    ax.set_yticks(ticks=np.arange(-0.6, 0.21, 0.2))
    ax.set_yticklabels(labels=np.arange(-60, 21, 20), fontsize=labelsize)
    ax.grid(axis='y', ls='-', lw=0.4, color='#EBE9E0', zorder=0)
    
    # remove spines
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)
        # make ticks and spines slightly grey
        ax.spines[spine].set_color('0.05')
        ax.tick_params(axis='both', colors='0.05')
    # remove ticks from top and right spines
    ax.tick_params(top=False, bottom=False, right=False, direction='out', width=0.5)
    
    return ax

fig = plt.figure(figsize=(4, 6))
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.1)

ax1 = fig.add_subplot(gs[0, 0])
make_panel(ax=ax1, dep=dep, scenario='SSP1-26')
ax2 = fig.add_subplot(gs[1, 0])
make_panel(ax=ax2, dep=dep, scenario='SSP2-45')
ax3 = fig.add_subplot(gs[2, 0])
make_panel(ax=ax3, dep=dep, scenario='SSP5-34')
ax4 = fig.add_subplot(gs[3, 0])
make_panel(ax=ax4, dep=dep, scenario='SSP5-85')

fig.supylabel('Relative Change in Regional Atmospheric Hg Deposition (%)', x=0.0, fontsize=10)

# -------------------------------------
# Add legend
# relabel the legend labels
# -------------------------------------
relabeling_dict = {'AfricaMidEast':'Africa + Middle East', 
                   'Asia':'Asia',
                   'Europe':'Europe', 
                   'FmrUSSR':'Former USSR',
                   'NorthAm':'North America', 
                   'Oceania':'Oceania', 
                   'SouthAm':'South America'}

handles, previous_labels = ax1.get_legend_handles_labels()
new_labels = [relabeling_dict[i] for i in previous_labels]
ax4.legend(handles=handles, labels=new_labels, ncol=2, bbox_to_anchor=(0.45, -0.56), 
           loc='center', fontsize=8, frameon=False)

# remove xlabels for all but the bottom plot
for ax in [ax1, ax2, ax3]:
    plt.setp(ax.get_xticklabels(), visible=False)

# add letter label to upper left corner of each plot
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.text(0.02, 0.96, chr(97+i), transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
# add scenario label to lower left corner of each plot
for scenario, ax in zip(['SSP1-2.6', 'SSP2-4.5', 'SSP5-3.4', 'SSP5-8.5'], [ax1, ax2, ax3, ax4]):
    ax.text(0.02, 0.05, scenario, transform=ax.transAxes, fontsize=10,  ha='left', va='bottom')

plt.savefig('../data/figures/main_text/Fig_3_relative_regional_deposition_change_v2.pdf', format='pdf', bbox_inches='tight')