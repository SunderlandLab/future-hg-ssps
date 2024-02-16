# plot_5_legacy deposition trends
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# set plotting parameters
mpl.rcParams['xtick.direction'] = 'in' #'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.direction'] = 'in' #'in'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
# set minor tick length
mpl.rcParams['xtick.minor.size'] = 1.5
mpl.rcParams['ytick.minor.size'] = 1.5
# set minor tick width
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.minor.width'] = 0.5


print('--------------------------------------------------------------')
print('- LEGACY DEPOSITION TRENDS TO LAND - FIG 5                   -')
print('--------------------------------------------------------------')

fontsize = 8 # per AGU style guide

# make sure that data extend from 1950 - 2300
tmp = pd.read_csv('../data/output/annual_legacy_deposition_to_land.csv')
df = pd.DataFrame()
for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85', 'no future emissions']:
    tmp2 = tmp[['year', scenario]].copy()
    tmp2.rename(columns={'year':'Year', scenario:'legacy + natural deposition'}, inplace=True)
    tmp2['scenario'] = scenario
    df = pd.concat([df, tmp2])

# define plotting parameters
plt_dict = {'SSP1-26':{'color':"#0072BD", 'ls':'-'}, 
            'SSP2-45':{'color':"#D95319", 'ls':'-'}, 
            'SSP5-34':{'color':"#EDB120", 'ls':'-'}, 
            'SSP5-85':{'color':"#A2142F", 'ls':'-'}, 
            'no future emissions':{'color':'k', 'ls':':'}}

# define plotting function -- note that this is the same as the function defined in plot_S4_...py
def make_plot2(df, ax, var, add_legend=False, loc=(1.05,1), ncol=5, lw=3, s=8,
              scenarios=['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85', 'no future emissions'],
              labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP5-3.4', 'SSP5-8.5', None]):
    # `ax`  : matplotlib ax to plot to
    # `var` : str or list
    for (scenario, label) in zip(scenarios, labels):
        sel = df[df['scenario']==scenario]
        base = sel[sel['Year']<=2010].copy()
        sel = sel[sel['Year']>=2010]

        scale = 1 #1e-3 # Mg to Gg
        if type(var) == str:
            ax.plot(sel['Year'],  sel[var]*scale, c=plt_dict[scenario]['color'], 
            ls=plt_dict[scenario]['ls'], lw=lw, label=label, zorder=4)
            
        elif type(var) == list:
            ax.plot(sel['Year'],  sel[var].sum(axis=1)*scale, c=plt_dict[scenario]['color'], 
            ls=plt_dict[scenario]['ls'], lw=lw, label=label, zorder=4)
        
    if type(var) == str:
        ax.plot(base['Year'], base[var]*scale, lw=lw, c='darkgrey')
        ax.scatter(2010, base[base['Year']==2010][var]*scale, s=s, c='k', zorder=5)

    elif type(var) == list:
        ax.plot(base['Year'], base[var].sum(axis=1)*scale, lw=lw, c='darkgrey')
        ax.scatter(2010, base[base['Year']==2010][var].sum(axis=1)*scale, s=s, c='k', zorder=5)

    start_yr = 1985
    ax.grid(c='whitesmoke', zorder=0)
    ax.set_xlim(start_yr, 2300)
    ax.set_xticks(np.arange(2000, 2301, 50))

    if add_legend != False:
        legend = ax.legend(ncol=ncol, loc='upper right', fancybox=False, facecolor='#fffff8', edgecolor='0.9', framealpha=1, fontsize=fontsize)
        legend.get_frame().set_linewidth(0.)

fig = plt.figure(figsize=(4.,2.5))
ax1 = plt.subplot(111)

make_plot2(df=df, ax=ax1, var='legacy + natural deposition', add_legend=True, loc='upper right', ncol=1, lw=2, s=8)

ax1.set_ylabel(r'Hg Deposition (Mg a$^{-1}$)', fontsize=fontsize+2)
# set ticklabel fontsize to 8 points
ax1.tick_params(axis='both', which='major', labelsize=fontsize)

ax1.set_ylim(400, 1600)

ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))

plt.tight_layout()
plt.savefig('../data/figures/main_text/Fig_5_legacy_and_natural_deposition_to_land.pdf', format='pdf', bbox_inches='tight')
