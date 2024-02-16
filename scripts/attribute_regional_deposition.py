import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# read in IFs -- these have not been normalized (e.g., reflect 500 Mg emissions)
IFs = xr.open_mfdataset('../data/dataverse/IFs/IFs.nc')

# define mini function to calculate emission magnitude from a given region
def get_emission_Mg(region_mask:str, species:str, year:int, emission_masks:xr.Dataset, emissions:xr.Dataset):
    '''
    This function calculates emission magnitude from a given region.
    
    Parameters
    ----------
    region_mask : str
        Name of region mask
    species : str
        Name of species
    year : int
        Year of emissions
    emission_masks : xr.Dataset
        Dataset containing emission masks
    emissions : xr.Dataset
        Dataset containing emissions for a given year and scenario
    
    Returns
    -------
    emission_magnitude : float
        Emission magnitude from a given region
    '''
    sec_per_yr = 60*60*24*365.25
    return (emission_masks[region_mask]*emissions[species].sel(time=year)*emission_masks['area']*1e6*sec_per_yr*1e-3).sum().values.item()

# define function to construct primary deposition from IFs and emissions
def construct_primary_deposition(emissions:xr.Dataset, year:int, scenario:str, 
                                 regions=['AfricaMidEast','Asia','Europe','FmrUSSR','NorthAm','Oceania','SouthAm'],
                                 species_list=['Hg0','Hg2','HgP']):
    ''' 
    This function calculates primary deposition from IFs and emissions.
    
    Parameters
    ----------
    emissions : xr.Dataset
        Dataset containing emissions for a given year and scenario
    year : int
        Year of emissions
    scenario : str
        Name of scenario
    
    Returns
    -------
    primary_deposition : xr.Dataset
        Dataset containing primary deposition for a given year and scenario
    '''

    emission_masks = xr.open_mfdataset('../data/dataverse/ssp_emissions_gridded/L2_masks_025d.nc')
    emission_masks['NorthAm_mask'] = emission_masks['USA_mask']+emission_masks['Canada_mask']+emission_masks['CentralAm_mask']

    # calculate emission magnitudes for each region and species
    regional_emission_magnitudes = {}
    for region in regions:
        regional_emission_magnitudes[region] = {}
        for species in species_list:
            tmp = region, get_emission_Mg(region_mask=region+'_mask', species=species, year=year, emission_masks=emission_masks, emissions=emissions)
            regional_emission_magnitudes[region][species] = tmp[1]

    # calculate primary deposition
    primary_deposition = (IFs['UNIFORM_Hg0'].copy()*0).rename('Primary_Deposition')
    for region in regions:
        for species in species_list:
            if species == 'HgP':
                primary_deposition += (IFs[f'{region}_Hg2']/500.)*regional_emission_magnitudes[region][species]
            else:
                primary_deposition += (IFs[f'{region}_{species}']/500.)*regional_emission_magnitudes[region][species]

    # create dataset to contain primary deposition
    primary_deposition = primary_deposition.to_dataset()
    primary_deposition = primary_deposition.assign_coords({"scenario": [scenario]})
    primary_deposition = primary_deposition.assign_coords({'time':[year]})
    primary_deposition['Primary_Deposition'].attrs = {'units':'μg m-2 yr-1','regions':regions, 'species':species_list}
    primary_deposition['time'].attrs = {'units':'year CE'}

    primary_deposition['Primary_Deposition'] = primary_deposition['Primary_Deposition'].expand_dims(dim='scenario')
    primary_deposition['Primary_Deposition'] = primary_deposition['Primary_Deposition'].expand_dims(dim='time')

    return primary_deposition

def get_legacy_deposition(IFs, legacy_emissions, year, scenario,
                          key_dict = {'waste_volatilization':'LAND_Hg0', 
                                      'terrestrial_emissions':'LAND_Hg0', 
                                      'ocean_evasion':'OCEAN_Hg0',
                                      'subaerial_volcanism':'UNIFORM_Hg0',}):
    '''
    This function calculates legacy deposition from IFs and emissions.
    
    Parameters
    ----------
    IFs : xr.Dataset
        Dataset containing influence functions
    legacy_emissions : pandas DataFrame
        DataFrame containing legacy emissions
    year : int
        Year of emissions
    scenario : str
        Name of scenario
    key_dict : dict
        Dictionary mapping legacy emission types to IFs
    
    Returns
    -------
    deposition : xr.Dataset
        Dataset containing legacy deposition for a given year and scenario
    '''

    # subset legacy emissions for a given year and scenario
    emissions = legacy_emissions[(legacy_emissions['Year']==year) & (legacy_emissions['scenario']==scenario)].copy()
    # add geogenic input to atmosphere
    emissions['subaerial_volcanism'] = 230. # Mg/yr (Geyman et al., 2023)
    # calculate ocean net evasion
    emissions['ocean_net_evasion'] = emissions['ocean_evasion'] - emissions['ocean_Hg0_uptake']

    for key, value in key_dict.items():
        # calculate deposition as product of IF and emission magnitude
        tmp = (IFs[value]*emissions[key].values)/500. # 500 is the emission magnitude in Mg used in IF simulations
        # check if variable "deposition" has been defined
        if 'deposition' in locals(): 
            deposition += tmp
        else: 
            deposition = tmp
        
    deposition = deposition.expand_dims('time')
    deposition = deposition.expand_dims('scenario')
    deposition = deposition.assign_coords(time=('time', [year]))
    deposition = deposition.assign_coords(scenario=('scenario', [scenario]))

    lat = np.hstack((-89.5, np.arange(-88., 90., 2.), 89.5))
    lon = np.arange(-180.,180.,2.5)
    deposition = xr.Dataset({'Legacy_Deposition': xr.DataArray(
                    data   = deposition,
                    dims   = ['time','scenario','lat','lon'],
                    coords = {'time': [year], 'scenario': [scenario], 'lat': lat, 'lon': lon},
                    attrs  = {'_FillValue': np.nan, 'units': 'μg m-2 yr-1'})})
    
    return deposition

def get_source_attributable_deposition(source_region:str, year:int, scenario:str, IFs=IFs, ):
    # ------------------------------------------------------------------------
    # --- load primary emissions
    if year == 2010:
        primary_emissions = xr.open_mfdataset('../data/dataverse/ssp_emissions_gridded/baseline_L2_025d_2010.nc')
    else:
        primary_emissions = xr.open_mfdataset(f'../data/dataverse/ssp_emissions_gridded/{scenario}_L2_025d_2020_2200.nc')

    primary_emissions = primary_emissions.sel(time=primary_emissions.time.dt.year.isin([year]))
    primary_emissions['time'] = primary_emissions['time'].dt.year

    # --- load legacy emissions
    legacy_emissions = pd.read_csv('../data/box_model/output/evasion_trends.csv')
    legacy_emissions['Year'] = legacy_emissions['Year'].astype(int)
    legacy_emissions = legacy_emissions[(legacy_emissions['scenario']==scenario) & (legacy_emissions['Year']==year)]
    # ------------------------------------------------------------------------

    # -- construct primary deposition for selected subset of {regions, species}
    # check if source region is in the list of anthropogenic regions
    if source_region in ['AfricaMidEast','Asia','Europe','FmrUSSR','NorthAm','Oceania','SouthAm']:
        source_attributable_deposition = construct_primary_deposition(emissions=primary_emissions, year=year, scenario=scenario, 
                                                                      regions=[source_region], species_list=['Hg0','Hg2','HgP'])

    elif source_region in ['waste_volatilization','terrestrial_emissions','ocean_evasion','subaerial_volcanism']:
        # dictionary for what IF pattern to use for each legacy emission type
        legacy_emission_patterns = {'waste_volatilization':'LAND_Hg0', 'terrestrial_emissions':'LAND_Hg0', 
                                    'ocean_evasion':'OCEAN_Hg0', 'subaerial_volcanism':'UNIFORM_Hg0'}
        emission_pattern = legacy_emission_patterns[source_region]
        source_attributable_deposition = get_legacy_deposition(IFs, legacy_emissions, year, scenario, key_dict = {source_region:emission_pattern})

    if 'area' not in source_attributable_deposition.coords:
        source_attributable_deposition = source_attributable_deposition.assign_coords({'area':IFs.area})

    return source_attributable_deposition

#test = get_source_attributable_deposition(source_region='Asia', year=2010, scenario='SSP1-26')

# get total deposition for each region
def get_regional_deposition_total_Mg(ds:xr.Dataset, masks:xr.Dataset, receptor_region:str):
    # get total deposition for each region
    mask = masks[receptor_region]
    # multiply mask by deposition
    dep = ds*mask
    dep = (dep*dep['area']).sum()*1e-12
    return dep.values.item()

# -- read in primary and legacy deposition
ds = xr.open_mfdataset('../data/dataverse/deposition_files/primary_and_legacy_deposition.nc').load()
ds['Total_Deposition'] = ds['Primary_Deposition']+ds['Legacy_Deposition']

receptor_masks = xr.open_mfdataset('../data/region_masks/receptor_regions_2x25_all.nc').load()
receptor_masks['NorthAm'] = receptor_masks['USA']+receptor_masks['Canada']+receptor_masks['CentralAm']
receptor_masks = receptor_masks.drop_vars(names=['USA','Canada','CentralAm'])

def attribute_year_and_scenario(scenario:str, year:int, ds=ds, receptor_masks=receptor_masks):

    primary_sources = ['AfricaMidEast','Asia','Europe','FmrUSSR','NorthAm','Oceania','SouthAm']
    natural_and_legacy_sources = ['waste_volatilization','terrestrial_emissions','ocean_evasion','subaerial_volcanism']
    source_regions = (primary_sources+natural_and_legacy_sources)

    output = {'scenario':[], 'year':[], 'receptor_region':[], 
            'source_region':[], 'receptor_deposition_total':[], 
            'source_attributable_deposition':[], 'units':[]}

    for receptor_region in receptor_masks:
        for source_region in source_regions:
            if source_region in primary_sources:
                dep_var = 'Primary_Deposition'
            elif source_region in natural_and_legacy_sources:
                dep_var = 'Legacy_Deposition'
            global_deposition_from_source   = get_source_attributable_deposition(source_region=source_region, year=year, scenario=scenario)[dep_var]
            regional_deposition_from_source = get_regional_deposition_total_Mg(ds=global_deposition_from_source, masks=receptor_masks, receptor_region=receptor_region)
            regional_deposition_total = get_regional_deposition_total_Mg(ds=ds['Total_Deposition'].sel(time=year, scenario=scenario), masks=receptor_masks, receptor_region=receptor_region)
            output['scenario'].append(scenario)
            output['year'].append(year)
            output['receptor_region'].append(receptor_region)
            output['source_region'].append(source_region)
            output['receptor_deposition_total'].append(regional_deposition_total)
            output['source_attributable_deposition'].append(regional_deposition_from_source)
            output['units'].append('Mg/yr')

    output = pd.DataFrame(output)
    output['fraction_from_source'] = output['source_attributable_deposition']/output['receptor_deposition_total']

    def get_category(x):
        if x in primary_sources:
            out = 'primary'
        elif x in natural_and_legacy_sources:
            out = 'legacy and natural'
        else:
            out = np.nan
        return out
    
    output['source_category'] = output['source_region'].apply(lambda x: get_category(x))

    return output

# -------------------------------------
# Call attribute_year_and_scenario() for all {year, scenario}
# -------------------------------------
output = pd.DataFrame()
for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
    for year in np.arange(2010, 2110, 10):
        print(scenario, year)
        if len(output) == 0:
            output = attribute_year_and_scenario(scenario=scenario, year=year)
        else:
            output = pd.concat((output, attribute_year_and_scenario(scenario=scenario, year=year)))
    output.to_csv(f'../data/output/regional_deposition_attribution_{scenario}.csv', index=False)
output.to_csv('../data/output/regional_deposition_attribution.csv', index=False)

# -- make **annual** legacy deposition file for use in Fig. 5
receptor_masks = xr.open_mfdataset('../data/region_masks/receptor_regions_2x25_all.nc')
all_receptor_mask = receptor_masks['NorthAm']*0
for region in ['AfricaMidEast','Asia','Europe','FmrUSSR','NorthAm','Oceania','SouthAm']:
    all_receptor_mask += receptor_masks[region]

years = np.arange(1900, 2301, 1).tolist()#2301, 1).tolist()
output = {'year':years}
legacy_emissions = pd.read_csv('../data/box_model/output/evasion_trends.csv')
legacy_emissions['Year'] = legacy_emissions['Year'].astype(int)
for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85', 'no future emissions']:
    print(f'Calculating annual legacy deposition for: {scenario}')
    output[scenario] = []
    for year in years:
        #print(scenario, year)
        deposition = get_legacy_deposition(IFs, legacy_emissions, year, scenario, 
                      key_dict = {'waste_volatilization':'LAND_Hg0', 
                                  'terrestrial_emissions':'LAND_Hg0', 
                                  'ocean_evasion':'OCEAN_Hg0',
                                  'subaerial_volcanism':'UNIFORM_Hg0',})
        deposition = (deposition['Legacy_Deposition']*IFs['area']*all_receptor_mask*1e-12).sum().values.item()
        output[scenario].append(deposition)

output = pd.DataFrame(output).to_csv('../data/output/annual_legacy_deposition_to_land.csv', index=False)