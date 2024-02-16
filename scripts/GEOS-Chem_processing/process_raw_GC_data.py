
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import date
import json

# --------------
# DEFINE THE FUNCTIONS
# --------------

def load_df(path_root, var_names=[], time_slice=None):
    '''
    Description: a function to load a dataframe of GEOS-Chem diagnostics 
    
    Inputs: 
    
    path_root: a string designating a path to the diagnostics
    var_names: a list of diagnostic names, grouped over time using the *
    time_slice (optional): a list containing two strings representting the start and end slice dates
    
    Outputs: a merged xarray Dataset
    
    Example Usage: load_df(path_root = 'GC_Rundir/OutputDir/', 
                           var_names = ['GEOSChem.Budget*'], 
                           time_slice = ['2005-01-16T00:00:00.000000000', '2010-01-16T00:00:00.000000000'])
    
    '''
    # ensure type of var_names is list, even if given as string
    var_names = list(var_names)
    
    if len(var_names)==1:
        df = xr.open_mfdataset(path_root+var_names[0])
        
    elif len(var_names)>1:
        print(var_names)
        df = xr.open_mfdataset(path_root+var_names[0])
        
        for v_name in var_names[1:]:
            print(v_name)
            tmp_df = xr.open_mfdataset(path_root+v_name)
            df = xr.merge([df, tmp_df])
            print(f'completed merge of {v_name}')
    
    # now slice time if time_slice != None
    if time_slice != None:
        df = df.sel(time=slice(time_slice[0], time_slice[1]))
        print(f'completed timeslice from {time_slice[0]} to {time_slice[1]}')
    return df


def annual_average(df, time_slice=[0,0]):
    base.mean('time')#, keep_attrs=True)
    base = base.assign_attrs({'time_start':time_slice[0], 'time_end':time_slice[1],
                              'description':'simulation averaged over period time_start to time_end',
                              'source':'process_GC_output.py'})
    return df


def get_wet_dep(df):
    '''
    Purpose: aggregate wet deposition by depositing species-grouping (0, 2, P) and convert to standard 
             units of μg m-2 yr-1
    Notes: df expected to contain aggregated collections (e.g., 'WetLossLSHg2')
    '''
   
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    
    # first do Hg2
    df['WetDep_Hg2'] = ( df['WetLossLSHg2'].sum('lev')+df['WetLossConvHg2'].sum('lev') )/df['AREA'] # kg/m2/s
    df['WetDep_Hg2'] = df['WetDep_Hg2']*1e9*3.154e+7 # μg/m2/yr
    df['WetDep_Hg2'] = df['WetDep_Hg2'].assign_attrs({'Species': 'Divalent Mercury (non-particulate)',
                                                      'Deposition Type': 'Wet',
                                                      'units':   'μg m-2 yr-1',
                                                      'Notes':   'created in `process_GC_output.ipynb',
                                                      'Created': f'{d1}'})
    
    # now do HgP
    df['WetDep_HgP'] = ( df['WetLossLSHgP'].sum('lev')+df['WetLossConvHgP'].sum('lev') )/df['AREA'] # kg/m2/s
    df['WetDep_HgP'] = df['WetDep_HgP']*1e9*3.154e+7 # μg/m2/yr
    df['WetDep_HgP'] = df['WetDep_HgP'].assign_attrs({'Species': 'Particulate Mercury',
                                                      'Deposition Type': 'Wet',
                                                      'units':   'μg m-2 yr-1',
                                                      'Notes':   'created in `process_GC_output.ipynb',
                                                      'Created': f'{d1}'})
    # now do all wet deposition
    df['Total_Hg_Wet_Dep'] = df['WetDep_Hg2']+df['WetDep_HgP']
    df['Total_Hg_Wet_Dep'] = df['Total_Hg_Wet_Dep'].assign_attrs({'Species': 'All Hg Species',
                                                                  'Deposition Type': 'Wet',
                                                                  'units':   'μg m-2 yr-1',
                                                                  'Notes':   'created in `process_GC_output.ipynb',
                                                                  'Created': f'{d1}'})
    
    return df

def get_dry_dep_total(df):
    '''
    Notes: df expected to contain aggregated collections (e.g., 'DryDepHg2')
    '''
   
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    
    seconds_per_year = 3.154e+7
    cm2_per_m2 = 1e2*1e2 
    g_per_molec = 200.59*(1./6.022e23) # molar mass of Hg * avogadro's number
    
    # aggregate all dry deposition
    df['Total_Hg_Dry_Dep'] = df['DryDep_Hg0']+df['DryDepHg2']+df['DryDepHgP'] # molec. cm-2 s-1
    
    df['Total_Hg_Dry_Dep'] = df['Total_Hg_Dry_Dep']*cm2_per_m2*seconds_per_year*g_per_molec*1e6 # μg m-2 yr-1

    df['Total_Hg_Dry_Dep'] = df['Total_Hg_Dry_Dep'].assign_attrs({'Species': 'All Hg Species',
                                                                  'Deposition Type': 'Dry',
                                                                  'units': 'μg m-2 yr-1',
                                                                  'Notes': 'created in `process_GC_output.ipynb',
                                                                  'Created': f'{d1}'})
    
    return df

def get_air_sea_exchange(df):
    '''
    Notes: 
    '''
   
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    
    seconds_per_year = 3.154e+7
    cm2_per_m2 = 1e2*1e2 
    ug_per_kg = 1e9
    
    # aggregate all dry deposition
    df['Net_Ocean_Hg0_Uptake'] = df['FluxHg0fromOceanToAir']-df['FluxHg0fromAirToOcean'] # kg/s
    
    for v in ['FluxHg0fromOceanToAir', 'FluxHg0fromAirToOcean', 'Net_Ocean_Hg0_Uptake']:
        df[v] = (df[v]/df['AREA'])*seconds_per_year*ug_per_kg # convert kg s-1 to μg m-2 yr-1

        df[v] = df[v].assign_attrs({'Species': 'elemental mercury (gaseous)',
                                    'Deposition Type': 'air-sea exchange',
                                    'units': 'μg m-2 yr-1',
                                    'Notes': 'created in `process_GC_output.ipynb',
                                    'Created': f'{d1}'})
    
    return df

def check_and_remove_empty_files(directory_path):
    '''
    Description: Code snippet for cleaning model output from simulations
                 which have not yet completed.
                 This script removes pesky diagnostics files created at the 
                 beginning of a model month which have not yet had data 
                 written to them. They are <150 Kb or 441 Kb, depending on 
                 collection. 
    Usage Notes: Note that this script used to delete all `MercuryOcean`
                 collection files (not just trailing month). I have added an
                 additional condition to avoid this (condition_0), but have 
                 not tested that it works
    '''
    import os

    for f in os.listdir(directory_path):
        file_size = os.path.getsize(directory_path+f)
        condition = ( (f.split('.')[0]=='GEOSChem') & (f.split('.')[-1]=='nc4') )
        condition_0 = (f.split('.')[1]!='MercuryOcean') # add condition to avoid deleting OceanMercury diag collection
        
        if ( (condition) & (file_size<50*1e3) ):
            print(f'removing {f} (filesize, Bytes: {file_size})')
            os.remove(directory_path+f)        
        
        elif ( (condition_0) & (condition) & (file_size<150*1e3) ):
            print(f'removing {f} (filesize, Bytes: {file_size})')
            os.remove(directory_path+f)

        elif ( (condition) & (file_size>400*1e3) & (file_size<500*1e3)):
            print(f'removing {f} (filesize, Bytes: {file_size})')
            os.remove(directory_path+f)
    
    return

# --------------
# DO THE STUFF
# --------------

## standard call sequence in terminal:
# conda activate GEOS-Chem
# cd /Users/bengeyman/Documents/MATS_Retro/Scripts_GC_Output/
# python process_raw_GC_data.py '/Volumes/LaCie/MATS_Retro/Model_Data/global_coarse_1990_all/' ['2016-01-16T00:00:00.000000000', '2019-01-16T00:00:00.000000000']

## note: zsh throws a "bad pattern" error when passing dates in square brackets. For now, it appears that modifying call to the below fixes the call:
# python process_raw_GC_data.py '/Volumes/LaCie/MATS_Retro_2/Model_Data/EGU_baseline/' '2016-01-16T00:00:00.000000000', '2019-01-16T00:00:00.000000000'


if __name__ == '__main__':
    from sys import argv # pass list of arguments
    path_root = argv[1]  # should look like '/Volumes/LaCie/MATS_Retro/Model_Data/global_coarse_1990_all/OutputDir/'
    time_slice1 = argv[2] # should look like: '2016-01-16T00:00:00.000000000'
    time_slice2 = argv[3] # should look like: '2019-01-16T00:00:00.000000000'
    
    time_slice = [time_slice1, time_slice2]
    
    ## Code section to remove diagnostics files from incomplete model runs
    #  this prevents xarray from throwing an error when it tries to open
    #  an empty NetCDF file
    delete_trailing_diagnostics_files = True
    if delete_trailing_diagnostics_files == True:
        check_and_remove_empty_files(directory_path=f'{path_root}/OutputDir/')
    
    
    # load model output and slice time
    base = load_df(path_root = path_root,
                   var_names=['/OutputDir/GEOSChem.WetLoss*',
                              '/OutputDir/GEOSChem.StateMet*',
                              '/OutputDir/GEOSChem.SpeciesConc*',
                              '/OutputDir/GEOSChem.DryDep*',
                              '/OutputDir/GEOSChem.MercuryEmis*',
                              '/OutputDir/GEOSChem.MercuryOcean*',
                              #'/OutputDir/GEOSChem.Budget*',
                             ],
                   time_slice=time_slice)
    print('completed load')
    
    # subset only the time indices where data are written
    # all first of the month values are empty for dry dep, wet dep, species conc
    # removing them first prevents including empty fields in averages
    # *this is not currently necessary, so is commented out*
    #base = base.where(base.time.dt.day != 1, drop=True)
    
    # take annual average
    base = base.mean('time')
    print('averaged over time dimension')

    # Start with a list of all the variables
    # to save space, I have saved a list of all of the variables to the following location
    var_list = pd.read_csv('/Users/bengeyman/Documents/Streets_Scenario_Modeling/Scripts/Data/GC_data_vars.csv')['Variable Name'].tolist()

    result = [] # initialize empty list to contain subset variables
    collections_list = ['DryDep_',  'SpeciesConc_',
                        'WetLossConv_', 'WetLossLS_',] # actual collections

    ancillary_list = ['Met_PRECTOT', 'Met_FROCEAN', 'Met_FRLAND','Met_FRLAKE', # specific vars to tag on
                      'Met_AIRVOL', 'Met_AIRDEN','AREA', 'EmisHg2HgPanthro', 'EmisHg0anthro',
                      'FluxHg0fromOceanToAir', 'FluxHg0fromAirToOcean']
    
    # load attributes (metadata) for ancillary variables
    with open('/Users/bengeyman/Documents/Streets_Scenario_Modeling/Scripts/Data/GC_ancillary_variable_attributes.json', 'r') as f:
        # Reading from json file
        ancillary_list_attrs = json.load(f)

    # do collections list first           
    for coll in collections_list:
        tmp_result = [i for i in var_list if i.startswith(coll)]
        result = result+tmp_result
    # add some extra useful variables                
    for var in ancillary_list:
        tmp_result = [i for i in var_list if i.startswith(var)]
        result = result+tmp_result
        
    print(result)
    
    # subset base to only the useful variables in 'result'
    base = base[result]

    for v in ancillary_list:
        base[v] = base[v].assign_attrs(ancillary_list_attrs[v])
    
    #base['EmisHg2HgPanthro'] = base['EmisHg2HgPanthro'].assign_attrs({'units':'kg/s'})
    #base['EmisHg0anthro'] = base['EmisHg0anthro'].assign_attrs({'units':'kg/s'})

    # now aggregate by species -- note: Hg
    
    # create list of Hg2 species to sum over
    Hg2_species_list = ['_HGCL2','_HGOHOH','_HGOHBRO','_HGOHCLO','_HGOHHO2',
                        '_HGOHNO2','_HGCLOH','_HGCLBR','_HGCLBRO','_HGCLCLO','_HGCLHO2','_HGCLNO2',
                        '_HGBR2','_HGBROH','_HGBRCLO','_HGBRBRO','_HGBRHO2','_HGBRNO2']

    aggregation_collections = ['DryDep', 'WetLossConv', 'WetLossLS']

    # aggregate Hg2
    for coll in aggregation_collections:
        for species in Hg2_species_list:
            if species == Hg2_species_list[0]:
                base[f'{coll}Hg2'] = base[f'{coll}{species}']
            else:
                base[f'{coll}Hg2'] += base[f'{coll}{species}']

            # now drop species which was just added
            base = base.drop_vars(f'{coll}{species}')

    HgP_species_list = ['_HG2ORGP','_HG2CLP']
    # aggregate HgP
    for coll in aggregation_collections:
        for species in HgP_species_list:
            if species == HgP_species_list[0]:
                base[f'{coll}HgP'] = base[f'{coll}{species}']
            else:
                base[f'{coll}HgP'] += base[f'{coll}{species}']

            # now drop species which was just added
            base = base.drop_vars(f'{coll}{species}')

    # Now do it all again for SpeciesConc

    Hg2_conc_species_list = ['_HGOH','_HGCL','_HGBR']+ Hg2_species_list
    
    # aggregate Hg2
    coll = 'SpeciesConc'
    for species in Hg2_conc_species_list:
        if species == Hg2_conc_species_list[0]:
            base[f'{coll}Hg2'] = base[f'{coll}{species}']
        else:
            base[f'{coll}Hg2'] += base[f'{coll}{species}']

        # now drop species which was just added
        base = base.drop_vars(f'{coll}{species}')

    HgP_conc_species_list = HgP_species_list + ['_HG2STRP']
    # aggregate HgP
    for species in HgP_conc_species_list:
        if species == HgP_conc_species_list[0]:
            base[f'{coll}HgP'] = base[f'{coll}{species}']
        else:
            base[f'{coll}HgP'] += base[f'{coll}{species}']

        # now drop species which was just added
        base = base.drop_vars(f'{coll}{species}')

    # CALL `get_wet_dep()`
    base = get_wet_dep(df = base)
    print('completed get wet dep')

    # CALL `get_dry_dep_total()`
    base = get_dry_dep_total(df = base)
    print('completed get dry dep')

    # CALL `get_air_sea_exchange()`
    base = get_air_sea_exchange(df = base)
    print('completed get air-sea exchange')
    
    base['Total_Hg_Dep'] = base['Total_Hg_Wet_Dep']+base['Total_Hg_Dry_Dep']
    base['Total_Hg_Dep'] = base['Total_Hg_Dep'].assign_attrs({'Deposition Type': 'Wet+Dry (no air-sea exchange)','units':'μg m-2 yr-1'})

    # SAVE base 
    base.to_netcdf(path_root+f'_GC_aggregated_{time_slice[0][0:4]}-{time_slice[1][0:4]}.nc')

