import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt_dict = {'SSP1-26':{'color':"#0072BD", 'ls':'-', 'label':'SSP1-2.6'}, 
            'SSP2-45':{'color':"#D95319", 'ls':'-', 'label':'SSP2-4.5'}, 
            'SSP5-34':{'color':"#EDB120", 'ls':'-', 'label':'SSP5-3.4'}, 
            'SSP5-85':{'color':"#A2142F", 'ls':'-', 'label':'SSP5-8.5'}, 
            }

df = pd.DataFrame()
for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
    Hg = pd.read_csv(f'../data/emissions_tabular/{scenario}.csv')
    #Hg = Hg[['Year','GLO_air_total']]
    Hg['Year'] = Hg['Year'].astype(int)
    Hg['scenario'] = scenario
    df = pd.concat([df, Hg])

# streets 2019b
df2 = pd.read_csv('../data/misc/Seventeen_World_Region_Total_Hg_Emissions_2000-2015.csv')
# transpose df2 and make first row the column names
df2 = df2.T
df2.columns = df2.iloc[0]
# drop first row
df2 = df2[1:]
# rename 'World region' column to 'Year'
df2.reset_index(inplace=True)
df2.rename(columns={'index':'Year'}, inplace=True)
df2['Year'] = df2['Year'].astype(int)

fig = plt.figure(figsize=(4,6))
ax = plt.subplot(111)
for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
    sel = df[df['scenario']==scenario].copy()
    ax.plot(sel['Year'], sel['GLO_air_total'], c=plt_dict[scenario]['color'], ls=plt_dict[scenario]['ls'], lw=2, marker='o', markeredgecolor='k', label=plt_dict[scenario]['label'], zorder=3)

ax.scatter(df2['Year'], df2['Global Total'], c='grey', edgecolor='k', label='Streets (2019b)', zorder=2)
ax.errorbar(df2['Year'], df2['Global Total'], yerr=[df2['Global Total'].values*0.2, df2['Global Total'].values*0.44] , fmt='none', lw=1.5, capsize=3, c='lightgrey', alpha=0.5, zorder=0)

ax.scatter([2009.8], [1960], marker='^', c='0.2', edgecolor='k', label='GMA 2013', zorder=3)
ax.errorbar([2009.8], [1960], yerr=[[1960*((1960-1010)/1960)], [1960*((4070-1960)/1960)]], fmt='none', lw=1.5, capsize=3, c='lightgrey', alpha=1, zorder=0)

ax.scatter([2014.8], [2220], marker='s', c='0.2', edgecolor='k', label='GMA 2018', zorder=3)
ax.errorbar([2014.8], [2220], yerr=[[2220*((2220-2000)/2220)], [2220*((2800-2000)/2220)]], fmt='none', lw=1.5, capsize=3, c='lightgrey', alpha=1, zorder=0)

# reduce pad between columns in legend
ax.legend(loc='lower right', ncols=2, fontsize=8, edgecolor='darkgrey', facecolor='#fffff8', fancybox=False);

ax.set_ylabel('Hg emissions (Mg a$^{-1}$)')
ax.set_xlim(2009, 2021)
ax.set_xticks(np.arange(2010, 2021, 5))
ax.set_xticks(np.arange(2009, 2022, 1), minor=True);
ax.set_ylim(0, 4350)
ax.set_yticks(np.arange(0, 4301, 1000))
ax.set_yticks(np.arange(0, 4301, 500), minor=True);

ax.text(s='Anthropogenic Hg Emissions to Air', x=2015, y=4220, ha='center', va='top', fontsize=10, color='0.2', fontweight='bold')
plt.savefig('../data/figures/supporting_information/Fig_S1_anthro_Hg_emissions_to_air_2010-2020.pdf', bbox_inches='tight')