import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('--------------------------------------------------------------')
print('- AREA-NORMALIZED NATURAL+LEGACY DEPOSITION PATTERNS - FIG 6-')
print('--------------------------------------------------------------')

fontsize = 8 # per AGU style guide

label_dict = {'AfricaMidEast':'AFM', 
              'Asia':'ASA', 
              'Europe':'EUR', 
              'FmrUSSR':'FSU', 
              'NorthAm':'NAM', 
              'Oceania':'OCA', 
              'SouthAm':'SAM',
              'OCEAN':'OCE', 
              'OTHER':'OTH', 
              'E_LAND':'Soil', 
              'E_OCEAN':'Ocean'}

def get_relative_deposition(emission_category='E_LAND'):
    anthro_regions = ['AfricaMidEast', 'Asia', 'Europe', 'FmrUSSR', 'NorthAm', 'Oceania','SouthAm']
    areas = pd.read_csv('../data/region_masks/region_mask_areas.csv')
    areas = areas[areas['region'].isin(anthro_regions)]

    anthro_receptors = ['D_'+i for i in anthro_regions]
    df = pd.read_csv('../data/output/Source-Receptor_Matrix_Hg0.csv')
    df = df[['Receptor', emission_category]] # subset to only contain emissions from 'emission_category'
    df = df[df['Receptor'].isin(anthro_receptors)].copy() # subset to only contain anthropogenic receptors
    df.loc[:, 'Receptor'] = df['Receptor'].str.replace('D_','') # remove 'D_' from receptor names
    df['Area'] = df['Receptor'].map(areas.set_index('region')['area [m2]']) # add area column
    df['areal_deposition'] = df[emission_category]/df['Area'] # calculate areal deposition
    average = df[emission_category].sum()/df['Area'].sum() # calculate area-weighted average

    df['rel_deposition'] = df['areal_deposition']/average # calculate relative deposition
    return df

def plot_relative_deposition(ax, df):
    x = np.arange(len(df))
    y = ((df['rel_deposition']-1)*100).values

    ax.bar(x=x, height=y, edgecolor='k', lw=0.5, facecolor='0.8', zorder=3)
    #plt.axhline(0, color='k', linestyle='-')
    ax.grid(ls='-', axis='y', c='0.95', zorder=1)
    ax.axhline(0, lw=1, color='k', linestyle='-', zorder=2)

    labels = [label_dict[i] for i in df['Receptor']]
    ax.set_xticks(x, labels, rotation=0);

    #ax.set_ylabel('Relative Deposition (%)', fontsize=fontsize);

    # expand y-axis by 10%
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))

    # add annotation for each bar
    for i in x:
        # get 2% of y-axis range
        y_offset = 0.02*(y_max-y_min)

        if y[i] < 0:
            ax.text(s=f'{int(np.round(y[i],0))}%', x=i, y=y[i]-y_offset, ha='center', va='top', fontsize=fontsize)
        else:
            ax.text(s=f'+{int(np.round(y[i],0))}%', x=i, y=y[i]+y_offset, ha='center', va='bottom', fontsize=fontsize)
        
    return

plt.figure(figsize=(7.5,3.25), facecolor='white')

ax = plt.subplot(121)
df = get_relative_deposition(emission_category='E_LAND')
plot_relative_deposition(ax, df)

# add label and title in upper left
ax.text(0.05, 0.95, 'a', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, c='0.0', va='top', ha='left')
ax.text(0.15, 0.95, 'Land Emissions', fontsize=fontsize+4, transform=ax.transAxes, c='0.0', va='top', ha='left')

ax.set_ylim(-35, 60)
ax.set_yticks(ticks=np.arange(-30, 61, 15))
ax.set_yticks(ticks=np.arange(-30, 61, 5), minor=True) # set minor yticks to 5
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_ylabel('Relative Deposition (%)', fontsize=fontsize+2)

ax = plt.subplot(122)
df = get_relative_deposition(emission_category='E_OCEAN')
plot_relative_deposition(ax, df)
# add label and title in upper left
ax.text(0.05, 0.95, 'b', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, c='0.0', va='top', ha='left')
ax.text(0.15, 0.95, 'Ocean Emissions', fontsize=fontsize+4, transform=ax.transAxes, c='0.0', va='top', ha='left')

ax.set_ylim(-35, 60)
ax.set_yticks(ticks=np.arange(-30, 61, 15))
ax.set_yticks(ticks=np.arange(-30, 61, 5), minor=True) # set minor yticks to 5
ax.tick_params(axis='both', which='major', labelsize=fontsize)

# --
plt.tight_layout()
plt.savefig('../data/figures/main_text/Fig_6_area_normalized_legacy_deposition.pdf', format='pdf', bbox_inches='tight')