import pandas as pd
import matplotlib.pyplot as plt

print('--------------------------------------------------------------')
print('- TRENDS IN OWN-SOURCE DEPOSITION - FIG S1 -------------------')
print('--------------------------------------------------------------')

# -- load regional deposition attribution data
df = pd.read_csv('../data/output/regional_deposition_attribution.csv')

output = pd.DataFrame() # empty dataframe to store results

# define anthropogenic receptor regions
anthro_receptors = ['AfricaMidEast','Asia','Europe','FmrUSSR','NorthAm','Oceania','SouthAm']

for source_region in anthro_receptors:
    source_dep_total = df[df['source_region']==source_region].groupby(['scenario','year'], as_index=False)['source_attributable_deposition'].sum()
    source_dep_self = df[(df['source_region']==source_region) & (df['receptor_region']==source_region)].groupby(['scenario','year'], as_index=False)['source_attributable_deposition'].sum()
    tmp = pd.merge(source_dep_total, source_dep_self, on=['scenario','year'], suffixes=('_total','_self'))
    tmp['fraction_self'] = tmp['source_attributable_deposition_self']/tmp['source_attributable_deposition_total']
    tmp['source_region'] = source_region
    output = pd.concat((output, tmp))

# calculate weighted-average 
output_agg = output.groupby(by=['scenario','year'], as_index=False)[['source_attributable_deposition_total','source_attributable_deposition_self']].sum()
output_agg['fraction_self'] = output_agg['source_attributable_deposition_self']/output_agg['source_attributable_deposition_total']

sel = output_agg
mean_2010 = sel[sel['year']==2010]['fraction_self'].mean()
mean_2100 = sel[sel['year']==2100]['fraction_self'].mean()

min_2010 = sel[sel['year']==2010]['fraction_self'].min()
min_2100 = sel[sel['year']==2100]['fraction_self'].min()

max_2010 = sel[sel['year']==2010]['fraction_self'].max()
max_2100 = sel[sel['year']==2100]['fraction_self'].max()

print('Fraction of anthropogenic deposition to region of origin')
print(f'2010: {mean_2010:.2f} ({min_2010:.2f} - {max_2010:.2f})')
print(f'2100: {mean_2100:.2f} ({min_2100:.2f} - {max_2100:.2f})')

# -- Fig S1. Trends in the fraction of anthropogenic emissions redepositing to region of origin

# --------------------------------------------------------------------
# Description:
# This script is used to generate the figure showing the trends in
# the fraction of regional emissions which re-deposit to the region of 
# origin.
#  -- 
# *lines* represent regional maxima and minima; *points* represent the regional
# deposition fraction for all anthropogenic emissions
# -- 
# Last Modified: Thur 15 Feb 2024
# --------------------------------------------------------------------

fontsize=8 # per AGU style guide

weighted_avg = output[output['year'].isin([2010, 2100])]
weighted_avg = weighted_avg.groupby(by=['year','scenario'], as_index=False)[['source_attributable_deposition_total','source_attributable_deposition_self']].sum()
weighted_avg['fraction_self'] = weighted_avg['source_attributable_deposition_self']/weighted_avg['source_attributable_deposition_total']

default_dict = {'facecolor':'0.2','linecolor':'0.2', 'lw':3, 'linealpha':0.9, 'facealpha':1.0, 'label':'Baseline'}
def add_point2(ax, y:int, year:int, scenario:str, output=output, plt_dict=default_dict):
    weighted_avgs = output.groupby(by=['year','scenario'], as_index=False)[['source_attributable_deposition_total','source_attributable_deposition_self']].sum()
    weighted_avgs['fraction_self'] = weighted_avgs['source_attributable_deposition_self']/weighted_avgs['source_attributable_deposition_total']

    x_avg = weighted_avgs[(weighted_avgs['scenario']==scenario) & (weighted_avgs['year']==year)]['fraction_self']

    output_sel = output[(output['scenario']==scenario) & (output['year']==year)]['fraction_self']
    ax.errorbar([x_avg], [y], xerr=(x_avg-output_sel.min(),output_sel.max()-x_avg),
                markerfacecolor=plt_dict['facecolor'],
                markeredgecolor='k', 
                alpha=plt_dict['facealpha'],
                marker='o',
                linewidth=plt_dict['lw'], 
                color=plt_dict['linecolor'], capsize=0,
                markersize=10, label=plt_dict['label'],
                zorder=3)
    # add text annotation of averge and range
    ax.text(x=output_sel.max()+0.02, y=y, s=year, c='0.3', ha='left', va='center', fontsize=9)

    return {'avg': x_avg, 'min': output_sel.min(), 'max':output_sel.max()}

plt_dicts = {'SSP1-26':{'y':1.,   'label':'SSP1-2.6', 'c':"#0072BD"},
             'SSP5-34':{'y':1.15, 'label':'SSP5-3.4', 'c':"#EDB120"},
             'SSP2-45':{'y':1.3,  'label':'SSP2-4.5', 'c':"#D95319"},
             'SSP5-85':{'y':1.45, 'label':'SSP5-8.5', 'c':"#A2142F"},
            }

values = {'year':[], 'scenario':[], 'avg':[], 'min':[], 'max':[], }

plt.figure(figsize=(4,5))
ax = plt.subplot(111)
# average over all regions
tmp = add_point2(ax, y=0.93, year=2010, scenario='SSP1-26', output=output, plt_dict=default_dict)

ax.fill_betweenx(y=[0,3], x1=tmp['min'], x2=tmp['max'], color='0.97')
ax.axvline(tmp['avg'].iloc[0], ls='-', c='grey', lw=1, zorder=1)

values['year'].append(2010)
values['scenario'].append('baseline')
values['avg'].append(tmp['avg'].iloc[0])
values['min'].append(tmp['min'])
values['max'].append(tmp['max'])

labels = []
for year, alpha, offset in zip([2030, 2050, 2070, 2100], [0.7, 0.85, 1.0, 1.0], [0.0, 0.03, 0.06, 0.09]):
     for scenario in ['SSP1-26','SSP5-34','SSP2-45','SSP5-85']:
        val_dict = plt_dicts[scenario]
        val_dict['linealpha'] = alpha
        default_dict['linecolor'] = val_dict['c']
        default_dict['facecolor'] = val_dict['c']
        default_dict['label'] = val_dict['label']
        y = val_dict['y']
        tmp = add_point2(ax, y=y+offset, year=year, scenario=scenario, output=output, plt_dict = default_dict)
        labels.append(val_dict['label'])

        values['scenario'].append(scenario)
        values['year'].append(year)
        values['avg'].append(tmp['avg'].iloc[0])
        values['min'].append(tmp['min'])
        values['max'].append(tmp['max'])

# display x tick labels as percentages
ax.set_xlim(0.14,0.56)
ax.set_xticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
ax.set_xticklabels(['','20','','30','','40','','50',''], fontsize=fontsize)
ax.set_xlabel('Regional Deposition Fraction (%)', fontsize=fontsize)

# remove y ticks
ax.set_ylim(1.6, 0.88)
ax.set_yticks([])

for spine in ['right','top','left']:
     ax.spines[spine].set_visible(False)

# add text annotation with scenario label
for scenario in ['SSP1-26','SSP5-34','SSP2-45','SSP5-85']:
     y = plt_dicts[scenario]['y']
     s = plt_dicts[scenario]['label']
     c = 'k'
     ax.text(x=0.28, y=y-0.02, s=s, color=c, ha='right', va='center', fontsize=fontsize)
ax.text(x=0.28, y=0.93-0.02, s='Baseline', color='k', ha='right', va='center', fontsize=fontsize)

# save figure
plt.savefig('../data/figures/supporting_information/Fig_S2_own_source_deposition_trends.pdf', format='pdf', bbox_inches='tight')

# display table with values
tmp = pd.DataFrame(values)
print('--')
tmp[['avg','min','max']] = tmp[['avg','min','max']].round(3)
print(tmp[tmp['scenario'].isin(['baseline','SSP1-26'])])
print(' ')