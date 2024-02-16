import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import json

dep = xr.open_mfdataset('../data/dataverse/deposition_files/primary_and_legacy_deposition.nc')

fontsize = 8 # per AGU style guide
plt.rcParams.update({'font.size': fontsize})

title_dict = {'SSP1-26':'SSP1-2.6', 'SSP2-45':'SSP2-4.5', 'SSP5-34':'SSP5-3.4', 'SSP5-85':'SSP5-8.5'}

def get_delta(ds, year_start, year_end, var='Legacy_Deposition'):
    delta = ds.sel(time=year_end)[var] - ds.sel(time=year_start)[var]
    return delta

def add_plot(ax, ds, year_start, year_end, var='Legacy_Deposition', cmap='RdBu_r', levels=[], title=None):
    delta = get_delta(ds, year_start, year_end, var)
    if len(levels)>0:
        im = delta.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, add_colorbar=False)
    else:
        im = delta.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.coastlines()
    ax.gridlines(ls='dotted', lw=0.5, color='grey')
    ax.set_title(title, fontsize=fontsize+4)

    return im, ax

vmin, vmax, vstep = -9, 1+1e-10, 1
# load cmap from json
with open('../data/colormaps/BlueYellowRed_adjusted.json') as f:
    delta_dep_cmap = json.load(f)
delta_dep_cmap = delta_dep_cmap['colors']

len_cmap = len(delta_dep_cmap)
delta_dep_cmap = ListedColormap(delta_dep_cmap)

dep_levels = np.linspace(vmin, vmax, len_cmap+1)

# set up (2, 2) subplots with gridspec
fig = plt.figure(figsize=(6,4.5))
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

for i, j, scenario, label in zip([0,0,1,1], [0,1,0,1], ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85'], ['a', 'b', 'c', 'd']):
    ax = fig.add_subplot(gs[i, j], projection=ccrs.Robinson())
    im, ax = add_plot(ax, dep.sel(scenario=scenario), 2010, 2100, levels=dep_levels, cmap=delta_dep_cmap, title=title_dict[scenario])
    # add bold label to upper left corner of each subplot
    ax.text(-0.0, 1.0, s=label, transform=ax.transAxes, size=fontsize+4, weight='bold')

# Adjust the location of the subplots on the page to make room for the colorbar
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.95, wspace=0.1, hspace=0.2)

# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.05])

# Draw the colorbar
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                    ticks=np.arange(-9, 2, 1),
                    shrink=0.7, label='Change in Legacy Hg Deposition, 2010 - 2100 ($\mathrm{\mu g \ m^{-2} \ a^{-1}}$)')
cbar.ax.minorticks_on()
cbar.ax.tick_params(labelsize=fontsize) 

plt.savefig('../data/figures/supporting_information/Fig_S4_legacy_deposition_change.png', dpi=500, bbox_inches='tight')
