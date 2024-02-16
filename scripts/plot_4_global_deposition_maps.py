import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

print('--------------------------------------------------------------')
print('- DEPOSITION MAPS - FIG 4 ------------------------------------')
print('--------------------------------------------------------------')

dep = xr.open_mfdataset('../data/dataverse/deposition_files/primary_and_legacy_deposition.nc')
dep['Total_Deposition'] = dep['Primary_Deposition']+dep['Legacy_Deposition']

def get_fractional_deposition_change(scenario:str, dep_var='Total_Deposition', comparison_year=2100, base_year=2010, dep=dep):
    # get fractional change in deposition from 2010 to 2050
    dep_base = dep[dep_var].sel(time=base_year, scenario=scenario)
    dep_comp = dep[dep_var].sel(time=comparison_year, scenario=scenario)
    frac_change = (dep_comp-dep_base)/dep_base
    return frac_change

scenario_labels = {'SSP1-26':'SSP1-2.6', 'SSP2-45':'SSP2-4.5', 'SSP5-34':'SSP5-3.4', 'SSP5-85':'SSP5-8.5'}

vmin, vmax, vstep = -0.6, 0.4+1e-10, 0.2

# create colormap with non-centered 0 point
cmap = plt.cm.get_cmap('RdBu_r', 250)
bottom_half = [cmap(i) for i in np.linspace(0,0.5,150)]
top_half    = [cmap(i) for i in np.linspace(0.5,0.9,100)]
colors = bottom_half + top_half
delta_dep_cmap = cmap.from_list('Custom cmap', colors, len(colors))

dep_levels = np.arange(vmin, vmax, 0.01)

def make_emission_panel(ax, ds, levels=dep_levels, cmap=cmap):
        ax.coastlines()
        ax.gridlines(ls='dotted', lw=0.5, color='grey')
        ds = ds.where(ds>0)

        im = np.log10(ds).plot(x='lon', y='lat', ax=ax, 
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap, levels=levels,
                        add_colorbar=False)
        return im, ax

def make_deposition_panel(ax, ds, levels=dep_levels, cmap=delta_dep_cmap):
        ax.coastlines()
        ax.gridlines(ls='dotted', lw=0.5, color='grey')
        im = ds.plot(x='lon', y='lat', ax=ax, 
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap, levels=levels,
                        add_colorbar=False)
        return im, ax   

subplot_n_matrix = np.array([[1,4,7,10],[2,5,8,11],[3,6,9,12]])
fig = plt.figure(figsize=(7,7))

n_rows, n_cols = 4, 3
for i, year in enumerate([2030, 2050, 2100]):
        for (subplt_n, scenario) in zip(subplot_n_matrix[i], ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']):
                delta_dep = get_fractional_deposition_change(scenario=scenario, dep_var='Total_Deposition', 
                                                             comparison_year=year, base_year=2010, dep=dep)
                ax = plt.subplot(n_rows, n_cols, subplt_n, projection=ccrs.Robinson())
                im, ax = make_deposition_panel(ax, delta_dep, levels=dep_levels, cmap=delta_dep_cmap)
                
                # annotate plots in first row with year
                if subplt_n in [1, 2, 3]:
                        ax.set_title(year, fontsize=14)
                else:
                        ax.set_title('')
                
                if subplt_n in [1, 4, 7, 10]:
                        ax.annotate(scenario_labels[scenario], xy=(-0.08, 0.5), xycoords='axes fraction', 
                                    rotation=90, fontsize=14, ha='center', va='center')

# Adjust the location of the subplots on the page to make room for the colorbar
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95,
                    wspace=0.1, hspace=0.2)

# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.02])

# Draw the colorbar
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                    ticks=np.arange(-0.6, 0.5, 0.2),
                    shrink=0.7, label='Relative Change in Total Hg Deposition (%)',)
cbar.ax.set_xticklabels(['-60', '-40', '-20', '0', '20', '40']);
cbar.ax.minorticks_on()

#plt.savefig('../data/figures/main_text/Fig_4_multipanel_deposition.png', dpi=1200, bbox_inches='tight')
#plt.savefig('../data/figures/main_text/Fig_4_multipanel_deposition.svg', format='svg', bbox_inches='tight')
plt.savefig('../data/figures/main_text/Fig_4_multipanel_deposition.pdf', format='pdf', bbox_inches='tight')