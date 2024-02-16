import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json

print('--------------------------------------------------------------')
print('- TRENDS IN PRIMARY VS. LEGACY DEPOSITION - FIG S2 -----------')
print('--------------------------------------------------------------')

# -- read regional deposition attribution data
df = pd.read_csv('../data/output/regional_deposition_attribution.csv')

# define function to get source-attributable deposition for a list of receptor regions under a given {scenario, year}
def get_dep(df=df, scenario=str, year=int, source_category=str, receptor_regions=list):
    return df[(df['scenario']==scenario) & 
              (df['year']==year) & 
              (df['source_category']==source_category) &
              (df['receptor_region'].isin(receptor_regions))]['source_attributable_deposition'].sum()

# Example Call:
#get_dep(df=df, scenario='SSP1-26', year=2010, source_category='legacy and natural', receptor_regions=['AfricaMidEast','Asia','Europe','NorthAm','Oceania','SouthAm','FmrUSSR'])

receptor_regions = ['AfricaMidEast','Asia','Europe','NorthAm','Oceania','SouthAm','FmrUSSR', 'OCEAN', 'OTHER']
anthro_receptors = ['AfricaMidEast','Asia','Europe','FmrUSSR','NorthAm','Oceania','SouthAm']
# delta primary (all-anthro regions)
ds = {'scenario':[], 'year':[], 'primary_dep':[], 'legacy_dep':[], 'total_dep':[], 'receptors':[]}
for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
    for year in np.arange(2010, 2110, 10):
        primary_dep = get_dep(df, scenario=scenario, year=year, source_category='primary', receptor_regions=anthro_receptors)
        legacy_dep = get_dep(df, scenario=scenario, year=year, source_category='legacy and natural', receptor_regions=anthro_receptors)
        total_dep = primary_dep+legacy_dep
        ds['scenario'].append(scenario)
        ds['year'].append(year)
        ds['primary_dep'].append(primary_dep)
        ds['legacy_dep'].append(legacy_dep)
        ds['total_dep'].append(total_dep)
        ds['receptors'].append(anthro_receptors)

ds = pd.DataFrame(ds)
ds.to_csv('../data/output/primary_legacy_deposition.csv', index=False)

# define function to get change between two timepoints for a given {scenario, variable}
def get_delta(ds, t0, t, scenario='SSP5-85', dep_var='total_dep'):
    ds = ds[ds['scenario']==scenario]
    d0 = ds[ds['year']==t0][dep_var].values.item()
    d  = ds[ds['year']==t][dep_var].values
    delta = d - d0
    return delta

# loop over years and scenarios and get change in deposition sources 
out_dict = {}
for dep_var in ['total_dep', 'primary_dep', 'legacy_dep', 'f_legacy_dep']:
    output = pd.DataFrame()
    for t in [2020, 2030, 2050, 2100]:
        row = {'year':[t],'dep_var':[dep_var]}
        for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
            if dep_var == 'f_legacy_dep':
                delta_legacy = get_delta(ds=ds, t0=2010, t=t, scenario=scenario, dep_var='legacy_dep')
                delta_total  = get_delta(ds=ds, t0=2010, t=t, scenario=scenario, dep_var='total_dep')
                delta = delta_legacy/delta_total
            elif dep_var in ['total_dep', 'primary_dep', 'legacy_dep']:
                delta = get_delta(ds=ds, t0=2010, t=t, scenario=scenario, dep_var=dep_var)
            row[scenario] = delta

        output = pd.concat((output, pd.DataFrame(row)))
    
    # print output
    if dep_var in ['total_dep','primary_dep','legacy_dep']:
        print(output.round(1))
    else:
        print(output.round(2))

dep_var = 'f_primary'
output = pd.DataFrame()
for t in [2030, 2050, 2100]:
    row = {'year':[t],'dep_var':[dep_var]}
    for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
        delta_total = get_delta(ds=ds, t0=2010, t=t, scenario=scenario, dep_var='total_dep')
        delta_primary = get_delta(ds=ds, t0=2010, t=t, scenario=scenario, dep_var='primary_dep')
        f_primary = delta_primary/delta_total
        if dep_var == 'f_primary':
            row[scenario] = f_primary
        else:
            row[scenario] = dep_var
    
    output = pd.concat((output, pd.DataFrame(row)))

# ---------------------------------------------------------------------------------

# -- open rates
with open('../data/colormaps/warm_gray.json', 'r') as f:
    cmap = json.load(f)
cmap = ListedColormap(cmap['colors'])

primary, legacy = np.meshgrid(np.arange(0, 2001, 100), np.arange(0, 2001, 100))
f_legacy = legacy/(primary+legacy)

def example_plot(ax, ds, scenario:str, hide_labels=False, plt_dict=None):

    ds = ds[ds['scenario']==scenario]
    x = np.array(ds['primary_dep'])
    y = np.array(ds['legacy_dep'])
    time = np.array(ds['year'])

    ax.scatter(x, y, facecolor='whitesmoke', edgecolor='k',
               marker='o', label=scenario, zorder=4, s=56)
    ax.scatter(x[0], y[0], facecolor='red', edgecolor='k',
               marker='o', label=scenario, zorder=4, s=56)
    ax.scatter(x[-1], y[-1], facecolor='blue', edgecolor='k',
               marker='o', label=scenario, zorder=4, s=56)
    
    ax.plot(x, y, color='k', alpha=0.5, zorder=2)
    
    ax.set_xlim(0,1300) 
    ax.set_ylim(1000,1600)
    ax.set_xticks(np.arange(0,1301,200))
    ax.set_yticks(np.arange(1000,1601,200))
    ax.tick_params(labelsize=10)
    if not hide_labels:
        ax.set_xlabel(r'Primary Deposition [Mg a$^{-1}$]', fontsize='x-large')

    im = ax.contourf(primary, legacy, f_legacy, 
                     levels=np.arange(0.4,1.001,0.001), 
                     vmin=0.4, vmax=1,
                     cmap=plt_dict[scenario]['cmap'], alpha=1, zorder=0)
    ax.contour(primary, legacy, f_legacy, 
               levels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               colors=['k'], linewidths=[0.7], alpha=0.15, zorder=0)
    return im

def add_vectors(ax, ds, scenario:str, plt_dict=None):

    ds = ds[ds['scenario']==scenario]
    initial_legacy = ds[ds['year']==2010]['legacy_dep'].values.item()
    initial_primary = ds[ds['year']==2010]['primary_dep'].values.item()
    delta_legacy = get_delta(ds=ds, t0=2010, t=2100, scenario=scenario, dep_var='legacy_dep').item()
    delta_primary  = get_delta(ds=ds, t0=2010, t=2100, scenario=scenario, dep_var='primary_dep').item()
    # -- 
    ls = '-'
    lw = 1
    # -- 
    if (float(delta_legacy) <= 0) & (float(delta_primary) <= 0):
        ax.plot([initial_primary, initial_primary+delta_primary,], [initial_legacy+delta_legacy, initial_legacy+delta_legacy], color='k', ls=ls, lw=lw, zorder=2)
        ax.plot([initial_primary, initial_primary],  [initial_legacy, initial_legacy+delta_legacy], color='k', ls=ls, lw=lw, zorder=2)       

    if (float(delta_legacy) >= 0) & (float(delta_primary) <= 0):
        ax.plot([initial_primary, initial_primary+delta_primary,], [initial_legacy, initial_legacy], color='k', ls=ls, lw=lw, zorder=2)
        ax.plot([initial_primary+delta_primary, initial_primary+delta_primary], [initial_legacy, initial_legacy+delta_legacy], color='k', ls=ls, lw=lw, zorder=2)
        ax.text(x=initial_primary+delta_primary, y=initial_legacy+delta_legacy, s=f'{delta_legacy:.0f}', ha='left', va='bottom', fontsize=10, color='k', zorder=4)

def annotate_change(ax, ds, scenario:str, display_text=True, print_text=True):
    ds = ds[ds['scenario']==scenario]
    delta_legacy   = int(np.round(get_delta(ds=ds, t0=2010, t=2100, scenario=scenario, dep_var='legacy_dep').item(),0))
    delta_primary  = int(np.round(get_delta(ds=ds, t0=2010, t=2100, scenario=scenario, dep_var='primary_dep').item(),0))
    delta_total    = int(np.round(delta_legacy+delta_primary))

    if delta_primary > 0:
        delta_primary = f'+{delta_primary}'
    if delta_legacy > 0:
        delta_legacy = f'+{delta_legacy}'
    if delta_total > 0:
        delta_total = f'+{delta_total}'
    if display_text:
        ax.text(x=0.6, y=0.2, 
                s=f'Primary: {delta_primary}\nLegacy:  {delta_legacy}\nTotal:    {delta_total}', 
                ha='left', va='center', fontsize=10, color='k', zorder=4, transform=ax.transAxes)
    if print_text:
        print(f'{scenario}: Primary: {delta_primary}, Legacy: {delta_legacy}, Total: {delta_total}')
    return 

plt_dict = {'SSP1-26':{'title':'SSP1-2.6', 'hide_labels':True, 'cmap':cmap}, 
            'SSP2-45':{'title':'SSP2-4.5', 'hide_labels':True, 'cmap':cmap}, 
            'SSP5-34':{'title':'SSP5-3.4', 'hide_labels':True, 'cmap':cmap}, 
            'SSP5-85':{'title':'SSP5-8.5', 'hide_labels':True, 'cmap':cmap}, }

# gridspec inside gridspec
fig = plt.figure(constrained_layout=True, figsize=(8, 3.78))

axs = fig.subplots(2, 2, sharex=False, sharey=False)

#fig.set_facecolor('0.95') # light gray background
for (ax, scenario, letter) in zip(axs.flatten(), ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85'], ['a','b','c','d']):
    pc = example_plot(ax, ds, scenario, hide_labels=plt_dict[scenario]['hide_labels'], plt_dict=plt_dict)
    # add label in upper left corner
    scenario_label = plt_dict[scenario]['title']
    ax.text(x=0.03, y=0.95, s=f'{letter}', ha='left', va='top', weight='bold', fontsize=12, color='k', zorder=4, transform=ax.transAxes)
    ax.text(x=0.10, y=0.95, s=f'{scenario_label}', ha='left', va='top', fontsize=12, color='k', zorder=4, transform=ax.transAxes)

    annotate_change(ax, ds, scenario, display_text=False)

cb = fig.colorbar(pc, ticks=np.arange(0.4,1.01,0.2),
                  label=None,
                  shrink=0.6, ax=axs, location='right',
                  pad=0.02)

cb.ax.tick_params(labelsize=10, ) 
fig.supylabel(r'Legacy + Natural Deposition [Mg a$^{-1}$]', fontsize='large')
fig.supxlabel('Primary Deposition [Mg a$^{-1}$]', fontsize='large')

plt.savefig('../data/figures/supporting_information/Fig_S3_legacy_vs_primary_dep_change.png', dpi=600, bbox_inches='tight')


