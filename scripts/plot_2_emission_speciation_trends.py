import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['xtick.minor.size'] = 1.5
mpl.rcParams['ytick.minor.size'] = 1.5
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.minor.width'] = 0.5

fontsize=8 # per AGU style guide

# -----------------------------------------------------------------
# Plot temporal changes in speciation of air emissions by scenario
# -----------------------------------------------------------------
def get_f_Hg0_air(df):
    from numpy import inf
    f_Hg0 = (df['GLO_air_Hg0']/df['GLO_air_total'])

    f_Hg0[f_Hg0 == inf] = np.nan  

    t = df['Year']
    f_Hg0 = pd.DataFrame({'Year':t, 'f_Hg0':f_Hg0})
    f_Hg0.replace([np.inf, -np.inf], np.nan, inplace=True)

    f_Hg0.loc[f_Hg0['f_Hg0']>=1, 'f_Hg0'] = np.nan
    f_Hg0.loc[f_Hg0['f_Hg0']<=0, 'f_Hg0'] = np.nan    

    return f_Hg0

plt_dicts = [{'label':'SSP5-8.5', 'c':"#A2142F"},
             {'label':'SSP2-4.5', 'c':"#D95319"},
             {'label':'SSP5-3.4', 'c':"#EDB120"},
             {'label':'SSP1-2.6', 'c':"#0072BD"}, ]

SSP1_26 = pd.read_csv('../data/emissions_tabular/SSP1-26.csv')
SSP2_45 = pd.read_csv('../data/emissions_tabular/SSP2-45.csv')
SSP5_34 = pd.read_csv('../data/emissions_tabular/SSP5-34.csv')
SSP5_85 = pd.read_csv('../data/emissions_tabular/SSP5-85.csv')

plt.figure(figsize=(4.,2.5))
for (df, plt_dict) in zip([SSP5_85, SSP2_45, SSP5_34, SSP1_26], plt_dicts):
    f_Hg0 = get_f_Hg0_air(df)
    f_Hg0 = f_Hg0[f_Hg0['Year']>=2010]
    plt.plot(f_Hg0['Year'], f_Hg0['f_Hg0'], 
            lw=2,
            marker='o', 
            markeredgecolor='k', 
            markerfacecolor=plt_dict['c'],
            color=plt_dict['c'],
            markeredgewidth=0.4, 
            markersize=5, 
            label=plt_dict['label'], zorder=3)
    
plt.grid(ls='-', lw=0.4, color='#EBE9E0', zorder=1)

legend = plt.legend(fontsize=fontsize, loc='upper right', fancybox=False, facecolor='#fffff8', edgecolor='0.9', framealpha=1)
legend.get_frame().set_linewidth(0.)

plt.xticks(ticks=np.linspace(2000,2250,6), labels=np.linspace(2000,2250,6).astype(int), )
# set xticklabels and yticklabels fontsize to 8 point
for label in (plt.gca().get_xticklabels()):
    label.set_fontsize(fontsize)
for label in (plt.gca().get_yticklabels()):
    label.set_fontsize(fontsize)
    
plt.xlim(2000, 2250)
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))

plt.yticks(ticks=np.linspace(0.1,0.7,7), labels=np.linspace(10,70,7).astype(int), )
plt.ylim(0.15,0.65)
plt.ylabel('Hg$^0$/ total Hg (%)', fontsize=fontsize+2)

plt.grid(ls='--')
plt.savefig('../data/figures/main_text/Fig_2_emission_speciation_trends.pdf', format='pdf', bbox_inches='tight')