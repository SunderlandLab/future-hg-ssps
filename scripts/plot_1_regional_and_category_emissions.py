import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

print('--------------------------------------------------------------')
print('- REGIONAL AND CATEGORY EMISSIONS - FIG 1 --------------------')
print('--------------------------------------------------------------')

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['xtick.minor.size'] = 1.5
mpl.rcParams['ytick.minor.size'] = 1.5
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.minor.width'] = 0.5

fontsize = 8 # per AGU style guide

# read colors from json files
with open('../data/colormaps/region_colors.json') as f:
    region_colors = json.load(f)
region_colors = region_colors['colors']

with open('../data/colormaps/category_colors.json') as f:
    category_colors = json.load(f)
category_colors = category_colors['colors']

regions = ['AFM','ASA','EUR','FSU','NAM','OCA','SAM']
category = 'air_total'

def is_empty(any_structure):
    # test if list or tuple is empty
    if any_structure:
        return False
    else:
        return True

def add_patch(ax, left, bottom, width, height, color, alpha=1, zorder=2):
    ''' Description: Adds patch to plot'''
    rect = plt.Rectangle((left, bottom), width, height, facecolor=color, alpha=alpha, zorder=zorder)
    ax.add_patch(rect)

def make_every_nth_blank(array, n):
    ''' Description: Makes every nth element of array blank. Used for xticks and yticks.
        Example: make_every_nth_blank(np.arange(0,2751,250), 2)'''
    array = array.astype(str)
    array[1::n] = ''
    return array

def round_n_significant_figs(x, n):
    ''' Description: Rounds x to n significant figures'''
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


# set x and y ticks
yticks      = np.arange(0,2751,250)
yticklabels = make_every_nth_blank(yticks, 2)
empty_yticklabels = np.repeat('', len(yticks))
xticks      = [2010, 2050, 2100, 2150, 2200, 2250]
xticklabels = xticks

def make_stackplot(ax, df, category, colors=[], xticks=xticks, xticklabels=xticklabels, 
                   yticks=yticks, yticklabels=yticklabels, legend=False, grid=True, title=''):
    x = df['Year']
    y = np.array([])
    for r in regions:
        tmp = df[f'{r}_{category}'].astype(float).values
        if len(y)==0:
            y = tmp
        else:
            y = np.vstack([y, tmp])
            
    labels = regions
    y_total = df[f'GLO_{category}'].astype(float).values

    plt.plot(x, y_total, lw=0.5, color='k', zorder=2)

    if not is_empty(colors):
        plt.stackplot(x, y, labels=labels, colors=colors, zorder=2)
    else:
        plt.stackplot(x, y, labels=labels, zorder=2)
    
    ax = plt.gca()
    
    plt.ylim(np.min(yticks), np.max(yticks))
    plt.yticks(ticks=yticks, labels=yticklabels, zorder=2, fontsize=fontsize) 
    
    plt.xlim(np.min(xticks), np.max(xticks))
    plt.xticks(ticks=xticks, labels=xticklabels, zorder=2, fontsize=fontsize)
    
    if legend!=False:
        plt.legend()
        
    if grid!=False:
        ax.yaxis.grid(ls='-', lw=0.5, color='#EBE9E0', zorder=1, alpha=0.5)
        ax.xaxis.grid(ls='-', lw=0.5, color='#EBE9E0', zorder=1, alpha=0.5)
    
    if title != '':
        plt.title(title, fontsize=fontsize+2)

emission_categories = ['Coal Combustion', 'Mining and Industry', 'Artisinal Gold Mining',  'Oil Combustion', 'Other']

def make_stackplot2(ax, df, colors=[], emission_categories=emission_categories,
                    xticks=xticks, xticklabels=xticklabels, yticks=yticks, yticklabels=yticklabels, 
                    legend=False, grid=True, title=''):
    x = df['Year']
    y = np.array([])
    for cat in emission_categories:
        tmp = df[cat].astype(float).values
        if len(y)==0:
            y = tmp
        else:
            y = np.vstack([y, tmp])
            
    labels = regions
    y_total = df['Total'].astype(float).values

    plt.plot(x, y_total, lw=0.5, color='k', zorder=2)

    if not is_empty(colors):
        plt.stackplot(x, y, labels=labels, colors=colors, zorder=2);
    else:
        plt.stackplot(x, y, labels=labels, zorder=2);
    
    ax = plt.gca()
    
    plt.ylim(np.min(yticks), np.max(yticks))
    plt.yticks(ticks=yticks, labels=yticklabels, zorder=2, fontsize=fontsize) 
    
    plt.xlim(np.min(xticks), np.max(xticks))
    plt.xticks(ticks=xticks, labels=xticklabels, zorder=2, fontsize=fontsize)
    
    if legend!=False:
        plt.legend()
        
    if grid!=False:
        ax.yaxis.grid(ls='-', lw=0.5, color='#EBE9E0', zorder=1, alpha=0.5)
        ax.xaxis.grid(ls='-', lw=0.5, color='#EBE9E0', zorder=1, alpha=0.5)
    
    if title != '':
        plt.title(title)

# -----------------------------------------------------------------
# Make (horizontally-arranged) stackplots of regional emissions by scenario
# -----------------------------------------------------------------
fig = plt.figure(figsize=(7.5, 4))
gs = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.2)

ax = fig.add_subplot(gs[0, 0])
SSP1_26 = pd.read_csv('../data/emissions_tabular/SSP1-26.csv')
make_stackplot(ax, SSP1_26[SSP1_26['Year']>=2010], category=category, 
               colors=region_colors, title='SSP1-2.6')
ax.set_xticklabels([])
ax.text(0.97, 0.97, 'a', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, va='top', ha='right')

ax = fig.add_subplot(gs[0, 1])
SSP2_45 = pd.read_csv('../data/emissions_tabular/SSP2-45.csv')
make_stackplot(ax, SSP2_45[SSP2_45['Year']>=2010], category=category, 
               yticklabels=empty_yticklabels,
               colors=region_colors, title='SSP2-4.5')
ax.set_xticklabels([])
ax.text(0.97, 0.97, 'b', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, va='top', ha='right')


ax = fig.add_subplot(gs[0, 2])
SSP5_34 = pd.read_csv('../data/emissions_tabular/SSP5-34.csv')
make_stackplot(ax, SSP5_34[SSP5_34['Year']>=2010], category=category, 
               yticklabels=empty_yticklabels,
               colors=region_colors, title='SSP5-3.4')
ax.set_xticklabels([])
ax.text(0.97, 0.97, 'c', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, va='top', ha='right')


ax = fig.add_subplot(gs[0, 3])
SSP5_85 = pd.read_csv('../data/emissions_tabular/SSP5-85.csv')
make_stackplot(ax, SSP5_85[SSP5_85['Year']>=2010], category=category, 
               yticklabels=empty_yticklabels,
               colors=region_colors, title='SSP5-8.5')
ax.set_xticklabels([])
ax.text(0.97, 0.97, 'd', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, va='top', ha='right')

# ------------------------------------------------------------------
# Add stackplots of category emissions by scenario
# ------------------------------------------------------------------

df = pd.read_csv('../data/emissions_tabular/Category_Emission_Trends.csv')

ax = fig.add_subplot(gs[1, 0])
SSP1_26 = df[df['Scenario']=='SSP1-26']
make_stackplot2(ax, SSP1_26, emission_categories=emission_categories,
                colors=category_colors)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=fontsize)
ax.text(0.97, 0.97, 'e', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, va='top', ha='right')

ax = fig.add_subplot(gs[1, 1])
SSP2_45 = df[df['Scenario']=='SSP2-45']
make_stackplot2(ax, SSP2_45, emission_categories=emission_categories,
                yticklabels=empty_yticklabels,
                colors=category_colors)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=fontsize)
ax.text(0.97, 0.97, 'f', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, va='top', ha='right')

ax = fig.add_subplot(gs[1, 2])
SSP5_34 = df[df['Scenario']=='SSP5-34']
make_stackplot2(ax, SSP5_34, emission_categories=emission_categories,
                yticklabels=empty_yticklabels,
                colors=category_colors)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=fontsize)
ax.text(0.97, 0.97, 'g', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, va='top', ha='right')

ax = fig.add_subplot(gs[1, 3])
SSP5_85 = df[df['Scenario']=='SSP5-85']
make_stackplot2(ax, SSP5_85, emission_categories=emission_categories,
                yticklabels=empty_yticklabels,
                colors=category_colors)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=fontsize)
ax.text(0.97, 0.97, 'h', fontsize=fontsize+4, weight='bold', transform=ax.transAxes, va='top', ha='right')

fig.supylabel(r'Hg Emissions (Mg a$^{-1}$)', fontsize=fontsize+2)

plt.savefig('../data/figures/main_text/Fig_1_regional_and_category_emissions.pdf', format='pdf')

# -----------------------------------------------------------------
# Calculate regional and temporal breakdown of emissions
# -----------------------------------------------------------------

def extract_time_snapshots(df, category='air_total'):
    from scipy import interpolate
    x_new = np.arange(1510, 2301, 1, dtype=int)
    df_out = pd.DataFrame({'Year':x_new})
    x = df.Year.values
    for v in df.columns[1:]:
        y = df[v].values
        f = interpolate.interp1d(x, y)
        y_new_tmp = f(x_new)
        df_out[v] = y_new_tmp

    out_dict = {}
    pie_dict1 = {}

    for start_yr, end_yr in zip([2010, 2050, 2100, 2200], [2049, 2099, 2199, 2300]):
        pie_dict1[f'{start_yr}-{end_yr}'] = []
        out_dict[f'{start_yr}-{end_yr}'] = {}
        t1 = ( (df_out['Year']<=end_yr) & (df_out['Year']>=start_yr) )
        tmp = df_out[t1]
        for r in regions:
            pie_dict1[f'{start_yr}-{end_yr}'].append(tmp[f'{r}_{category}'].sum())
            out_dict[f'{start_yr}-{end_yr}'][f'{r}_{category}'] = tmp[f'{r}_{category}'].sum()
            
    pie_dict2 = {}
    
    for r in regions:
        pie_dict2[f'{r}_{category}'] = []
        for start_yr, end_yr in zip([2010, 2050, 2100, 2200], [2049, 2099, 2199, 2300]):
            t1 = ( (df_out['Year']<=end_yr) & (df_out['Year']>=start_yr) )
            tmp = df_out[t1]
            pie_dict2[f'{r}_{category}'].append(tmp[f'{r}_{category}'].sum())

    return pie_dict1, pie_dict2, out_dict


# -----------------------------------------------------------------
# Define function for plotting pie charts of emission breakdown by region and time period
# function adapted from: https://matplotlib.org/stable/gallery/pie_and_polar_charts/nested_pie.html
# -----------------------------------------------------------------

def make_nested_pie_chart(ax, vals, outer_colors, inner_colors, title='', size=0.3, edgecolor='w', lw=0.1):
    if not is_empty(outer_colors):
        ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
                wedgeprops=dict(width=size, edgecolor=edgecolor, linewidth=lw), 
                startangle=270, counterclock=False)
    if not is_empty(inner_colors):
        ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
               wedgeprops=dict(width=size, edgecolor=edgecolor, linewidth=lw), 
               startangle=270, counterclock=False)
    ax.set(aspect="equal", title=title)
    return

def plot_pie_chart_demo():
    '''Function for testing make_nested_pie_chart() function'''
    plt.figure(figsize=(10,5))
    ax = plt.subplot(121)
    vals = np.array([pie_dict1['2010-2049'], pie_dict1['2050-2099'], 
                     pie_dict1['2100-2199'], pie_dict1['2200-2300'],])
    outer_colors = ['black', 'gray','lightgray','whitesmoke']
    inner_colors = region_colors
    make_nested_pie_chart(ax, vals, outer_colors, inner_colors, title='Top level: Time Periods')

    ax = plt.subplot(122)
    vals = np.array([pie_dict2['AFM_air_total'], pie_dict2['ASA_air_total'], 
                     pie_dict2['EUR_air_total'], pie_dict2['FSU_air_total'],
                     pie_dict2['NAM_air_total'], pie_dict2['OCA_air_total'],
                     pie_dict2['SAM_air_total'],
                    ])
    outer_colors = region_colors
    inner_colors = ['black', 'gray','lightgray','whitesmoke']
    make_nested_pie_chart(ax, vals, outer_colors, inner_colors, title='Top level: Regions')

# -----------------------------------------------------------------
# Make (simpler) pie charts of emission breakdown by time period
# -----------------------------------------------------------------
def get_E_total(pie_dict1, print_total=True, print_splits=True):
    total = 0
    for k in pie_dict1.keys():
        total+= np.array(pie_dict1[k]).sum()
    if print_total == True:
        print(f'Total (Gg): {np.round(total*1e-3,2)}')
    if print_splits == True:
        print('Splits (Gg):')
        for k in pie_dict1.keys():
            print(f'{k}: {np.round(np.array(pie_dict1[k]).sum()*1e-3,2)}')
    return total

plt.figure(figsize=(5, 1.25))

outer_colors = ['0.95', '0.90','0.85','0.8']
inner_colors = region_colors

ax = plt.subplot(141)
SSP1_26 = pd.read_csv('../data/emissions_tabular/SSP1-26.csv') 
pie_dict1, pie_dict2, out_dict = extract_time_snapshots(df=SSP1_26, category='air_total')
vals = np.array([pie_dict1['2010-2049'], pie_dict1['2050-2099'], pie_dict1['2100-2199'], pie_dict1['2200-2300'],])
make_nested_pie_chart(ax, vals, outer_colors=outer_colors, inner_colors=[], title='', size=0.4, edgecolor='k', lw=0.8)
total = get_E_total(pie_dict1, print_total=True)
ax.text(x=0., y=0.03, s=f'{int(round_n_significant_figs(total*1e-3, n=3))}', ha='center', fontsize=12, zorder=5)
ax.text(x=0., y=-0.4, s=f'Gg', ha='center', fontsize=12, zorder=5)

ax = plt.subplot(142)
SSP2_45 = pd.read_csv('../data/emissions_tabular/SSP2-45.csv') 
pie_dict1, pie_dict2, out_dict = extract_time_snapshots(df=SSP2_45, category='air_total')
vals = np.array([pie_dict1['2010-2049'], pie_dict1['2050-2099'], pie_dict1['2100-2199'], pie_dict1['2200-2300'],])
make_nested_pie_chart(ax, vals, outer_colors=outer_colors, inner_colors=[], title='', size=0.4, edgecolor='k', lw=0.8)
total = get_E_total(pie_dict1, print_total=True)
ax.text(x=0., y=0.03, s=f'{int(round_n_significant_figs(total*1e-3, n=3))}', ha='center', fontsize=12, zorder=5)
ax.text(x=0., y=-0.4, s=f'Gg', ha='center', fontsize=12, zorder=5)

ax = plt.subplot(143)
SSP5_34 = pd.read_csv('../data/emissions_tabular/SSP5-34.csv') 
pie_dict1, pie_dict2, out_dict = extract_time_snapshots(df=SSP5_34, category='air_total')
vals = np.array([pie_dict1['2010-2049'], pie_dict1['2050-2099'], pie_dict1['2100-2199'], pie_dict1['2200-2300'],])
make_nested_pie_chart(ax, vals, outer_colors=outer_colors, inner_colors=[], title='', size=0.4, edgecolor='k', lw=0.8)
total = get_E_total(pie_dict1, print_total=True)
ax.text(x=0., y=0.03, s=f'{int(round_n_significant_figs(total*1e-3, n=3))}', ha='center', fontsize=12, zorder=5)
ax.text(x=0., y=-0.4, s=f'Gg', ha='center', fontsize=12, zorder=5)

ax = plt.subplot(144)
SSP5_85 = pd.read_csv('../data/emissions_tabular/SSP5-85.csv') 
pie_dict1, pie_dict2, out_dict = extract_time_snapshots(df=SSP5_85, category='air_total')
vals = np.array([pie_dict1['2010-2049'], pie_dict1['2050-2099'], pie_dict1['2100-2199'], pie_dict1['2200-2300'],])
make_nested_pie_chart(ax, vals, outer_colors=outer_colors, inner_colors=[], title='', size=0.4, edgecolor='k', lw=0.8)
total = get_E_total(pie_dict1, print_total=True)
ax.text(x=0., y=0.03, s=f'{int(round_n_significant_figs(total*1e-3, n=3))}', ha='center', fontsize=12, zorder=5)
ax.text(x=0., y=-0.4, s=f'Gg', ha='center', fontsize=12, zorder=5)
plt.tight_layout()
plt.savefig('../data/figures/main_text/Fig_1_regional_emissions_insets.pdf', format='pdf')

# -----------------------------------------------------------------
# Make custom legend for figure
# -----------------------------------------------------------------
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ec = '0.2'
lw = 0.4
legend_elements = [Patch(facecolor=region_colors[0], edgecolor=ec, label='Africa + Middle East', lw=lw),
                   Patch(facecolor=region_colors[1], edgecolor=ec, label='Asia', lw=lw),
                   Patch(facecolor=region_colors[2], edgecolor=ec, label='Western Europe', lw=lw),
                   Patch(facecolor=region_colors[3], edgecolor=ec, label='Former USSR', lw=lw),
                   Patch(facecolor=region_colors[4], edgecolor=ec, label='North America', lw=lw),
                   Patch(facecolor=region_colors[5], edgecolor=ec, label='Oceania', lw=lw),
                   Patch(facecolor=region_colors[6], edgecolor=ec, label='South America', lw=lw),
                   Patch(facecolor='None', edgecolor='None', label='', lw=lw),]

legend_elements_2 = [Patch(facecolor=category_colors[0], edgecolor=ec, label='Coal Combustion', lw=lw),
                     Patch(facecolor=category_colors[1], edgecolor=ec, label='Mining + Industry', lw=lw),
                     Patch(facecolor=category_colors[2], edgecolor=ec, label='ASGM', lw=lw),
                     Patch(facecolor=category_colors[3], edgecolor=ec, label='Oil Combustion', lw=lw),
                     Patch(facecolor=category_colors[4], edgecolor=ec, label='Other', lw=lw),
                     ]

# Create the figure
fig = plt.figure(figsize=(7.5, 4))
ax = plt.subplot(211)
ax.legend(handles=legend_elements, loc='upper left', ncols=2, frameon=False, fontsize=fontsize-1)
ax.axis('off')
ax = plt.subplot(212)
ax.legend(handles=legend_elements_2, loc='lower left', ncols=2, frameon=False, fontsize=fontsize-1)
ax.axis('off')
fig.patch.set_visible(False)

plt.savefig('../data/figures/main_text/Fig_1_regional_and_category_emissions_legend.pdf', format='pdf', bbox_inches='tight')