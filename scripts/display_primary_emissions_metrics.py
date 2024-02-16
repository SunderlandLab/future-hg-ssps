import numpy as np
import pandas as pd

def get_cumulative_emissions(df, var='GLO_air_total'):
    ''' Description: reads emissions and returns the cumulative emissions using linear interpolation.'''
    df = df[df['Year']>=2010].copy()
    df = pd.merge(df, pd.DataFrame({'Year':np.arange(2010, 2301)}), how='outer', on='Year')
    df.sort_values(by='Year', inplace=True)
    df = df.reset_index(drop=True)
    df[var] = df[var].astype(float)
    df = df.interpolate()
    total_emission_Gg = df[var].sum()*1e-3
    return total_emission_Gg

print('--------------------------------------------------------------')
print('--- Abstract ')
print('--------------------------------------------------------------')
print(' ')
print('-- Cumulative emissions, by scenario (2010 - 2300) --------')
print(' ')
data_path = '../data/emissions_tabular/'
for scenario in ['SSP1-26','SSP2-45','SSP5-34','SSP5-85']:
    df = pd.read_csv(data_path+f'{scenario}.csv')
    print(f'Emissions for {scenario}')
    ALW = 0
    for v in ['GLO_air_total', 'GLO_LW_HgT']:
        total_emission_Gg = get_cumulative_emissions(df, var=v)
        print(f'{(v).ljust(13)} : {np.round(total_emission_Gg,1)} Gg')
        ALW += total_emission_Gg
    print(('ALW').ljust(13)+f' : {np.round(ALW,1)} Gg')
    print('- ')

print(' ')
print('-- Peak primary emissions, by scenario (2010 - 2300) ------')
print(' ')

v = 'GLO_air_total'
print(f'Variable: {v}')
for scenario in ['SSP1-26','SSP2-45','SSP5-34','SSP5-85']:
    df = pd.read_csv(data_path+f'{scenario}.csv')
    df = df[df['Year']>=2010].copy()
    idxmax = df[v].idxmax()
    year_max = df.loc[idxmax,"Year"]
    value_max = np.round(df.loc[idxmax,v],1)
    print(f'{scenario} : {value_max} Mg/yr in {year_max}')
print('--')

print(' ')
print('-- Land and water vs. air emissions comparison')
print(' ')
print('Displaying ratio of land + water emissions to air emissions')
print('--')
print('Baseline (2010)')
for scenario in ['SSP1-26','SSP2-45','SSP5-34','SSP5-85']:
    df = pd.read_csv(data_path+f'{scenario}.csv')
    df['LW_to_air_ratio'] = df['GLO_LW_HgT']/df['GLO_air_total']
    if scenario == 'SSP1-26':
        LW = np.round(df[df['Year']==2010]['GLO_LW_HgT'].item(),1)
        air = np.round(df[df['Year']==2010]['GLO_air_total'].item(),1)
        print(f'LW releases  : {LW} Mg/yr')
        print(f'Air releases : {air} Mg/yr')
        ratio = np.round(df[df['Year']==2010]['LW_to_air_ratio'].item(),1)
        print(f'Ratio : {ratio}')
        print('--')
        print('End Century (2100)')
    ratio = np.round(df[df['Year']==2100]['LW_to_air_ratio'].item(),1)
    print(f'{scenario} : {ratio}')
print('--')
print('Cumulative (2010 - 2300)')
for scenario in ['SSP1-26','SSP2-45','SSP5-34','SSP5-85']:
    df = pd.read_csv(data_path+f'{scenario}.csv')
    df = df[df['Year']>=2010].copy()
    LW = get_cumulative_emissions(df, var='GLO_LW_HgT')
    air = get_cumulative_emissions(df, var='GLO_air_total')
    ratio = np.round(LW/air,1)
    print(f'{scenario} : {ratio}')

print('--------------------------------------------------------------')
print('--- 3.1.2 Scenario-specific patterns in primary Hg anthropogenic releases')
print('--------------------------------------------------------------')
print(' ')
print('-- SSP 1-2.6')
print('')
df = pd.read_csv(data_path+'SSP1-26.csv')
# - regional contributions to air emissions in base year
region_display_names = {'AFM':'Africa + Middle East',
                        'ASA':'Asia',
                        'EUR':'Western Europe',
                        'FSU':'Former Soviet Union',
                        'NAM':'North America',
                        'OCA':'Oceania',
                        'SAM':'South America',}
tmp = df[df['Year']==2010].copy()
global_emissions_2010 = tmp['GLO_air_total'].item()
print(f'Regional contributions to 2010 global air emissions ({np.round(global_emissions_2010,1)} Mg/yr) --- ')
print(' ')
for region in region_display_names.keys():
    regional_emissions = tmp[f'{region}_air_total'].item()
    regional_contribution = np.round(regional_emissions/global_emissions_2010*100,1)
    print(f'{region_display_names[region].ljust(20)} : {(str(np.round(regional_emissions,1))).ljust(6)} Mg/yr ({regional_contribution}%)')
print(' ')

print('Air emissions in 2100 ----')
tmp = df[df['Year']==2100].copy()
global_emissions_2100 = tmp['GLO_air_total'].item()
SSP1_26_global_emissions_2100 = global_emissions_2100 # for later use
# calculate reduction in global emissions (2010 - 2100)
global_emissions_reduction = np.round( ((global_emissions_2010 - global_emissions_2100)/global_emissions_2010)*100, 1)
# identify region (column name) with highest air emissions
regional_air_cols = [f'{region}_air_total' for region in region_display_names.keys()]
highest_emitting_region_col = tmp.loc[:,regional_air_cols].idxmax(axis=1).item() # column name
highest_emitting_region = highest_emitting_region_col.split('_')[0]              # region code
highest_emitting_region_emissions = tmp[highest_emitting_region_col].item()      # emission magnitude
fraction_highest_emitting_region = np.round(highest_emitting_region_emissions/global_emissions_2100*100,1)
print(f'Global emissions : {np.round(global_emissions_2100,1)} Mg/yr')
print(f'Global emissions reduction (2010 - 2100) : {global_emissions_reduction}%')
print(f'Highest emitting region : {region_display_names[highest_emitting_region]}')
print(f'{region_display_names[highest_emitting_region]} emissions : {np.round(highest_emitting_region_emissions,1)} Mg/yr ({fraction_highest_emitting_region}% of total)')

print(' ')
print('-- SSP 2-4.5')
print(' ')
df = pd.read_csv(data_path+'SSP2-45.csv')
SSP2_45_global_emissions_2100 = df[df['Year']==2100]['GLO_air_total'].item()
# calculate percent difference between SSP2-4.5 and SSP1-2.6 emissions in 2100
percent_difference = np.round(((SSP2_45_global_emissions_2100/SSP1_26_global_emissions_2100)-1)*100,1)
print(f'Global emissions in 2100 : {np.round(SSP2_45_global_emissions_2100,1)} Mg/yr')
print(f'Emissions under SSP2-4.5 are {percent_difference}% higher than SSP1-2.6 in 2100')

print(' ')
print('-- SSP 5-3.4')
print(' ')
df = pd.read_csv(data_path+'SSP5-34.csv')
SSP5_34_global_emissions_2100 = df[df['Year']==2100]['GLO_air_total'].item()
print(f'Global emissions in 2100 : {np.round(SSP5_34_global_emissions_2100,1)} Mg/yr')

# calculate peak emissions for df
df = df[df['Year']>=2010].copy()
idxmax = df['GLO_air_total'].idxmax()
year_max = df.loc[idxmax,"Year"]
value_max = df.loc[idxmax,'GLO_air_total']
value_2010 = df[df['Year']==2010]['GLO_air_total'].item()
rel_change_pct = (value_max-value_2010)/value_2010*100
print(f'Peak emissions : {np.round(value_max,1)} Mg/yr in {year_max} ({np.round(rel_change_pct,1)}% increase from 2010)')

print(' ')
print('-- SSP 5-8.5')
print(' ')
df = pd.read_csv(data_path+'SSP5-85.csv')
SSP5_85_global_emissions_2100 = df[df['Year']==2100]['GLO_air_total'].item()
print(f'Global emissions in 2100 : {np.round(SSP5_85_global_emissions_2100,1)} Mg/yr')
emission_ratios = []
for value in [SSP1_26_global_emissions_2100, SSP2_45_global_emissions_2100, SSP5_34_global_emissions_2100]:
    emission_ratios.append(SSP5_85_global_emissions_2100/value)
ratio_min, ratio_max = min(emission_ratios), max(emission_ratios)
print(f'Emissions under SSP5-8.5 are {np.round(ratio_min,1)}x to {np.round(ratio_max,1)}x higher than other scenarios in 2100')

# calculate growth in AFM emissions between 2010 and peak
df = pd.read_csv(data_path+'SSP5-85.csv')
df = df[df['Year']>=2010].copy()
idxmax = df['AFM_air_total'].idxmax()
year_max = df.loc[idxmax,"Year"]
value_max = df.loc[idxmax,'AFM_air_total']
value_2010 = df[df['Year']==2010]['AFM_air_total'].item()
rel_change_pct = (value_max-value_2010)/value_2010*100
print(f'Africa + Middle East peak emissions : {np.round(value_max,1)} Mg/yr in {year_max} ({np.round(rel_change_pct,1)}% increase from 2010 value of {np.round(value_2010,1)} Mg/yr)')

print('--------------------------------------------------------------')
print('- 3.1.3 Hg emissions increasingly decoupled from CO2 emissions in the future ')
print('--------------------------------------------------------------')

def get_change(df, yr_start:int, yr_end:int, var:str):
    '''Calculate the change in a variable between a start year and an end year
        df: dataframe
        yr_start: start year
        yr_end: end year
        var: variable to calculate change in
    
        returns: change in variable between start and end year'''
    
    v_start = df[df['Year']==yr_start][var].item()
    v_end = df[df['Year']==yr_end][var].item()
    return (v_end-v_start)/v_start

def get_peak(df, var:str, yr_range=[2010, 2300]):
    '''Calculate the peak value of a variable between a start year and an end year
        df: dataframe
        var: variable to calculate peak in
        yr_range: range of years to calculate peak in
        
        returns: dictionary with peak year and peak value'''
    
    df = df[(df['Year']>=yr_range[0]) & (df['Year']<=yr_range[1])]
    peak_year = df[df[var]==df[var].max()]['Year'].item()
    peak_value = df[var].max()
    return {'peak_year':peak_year, 'peak_value':peak_value}

def get_relative_difference(val1, val2):
    '''Calculate the relative difference between two values'''
    return (val2-val1)/val1


def get_peak_vals(df, scenario:str):
    df = df[df['Scenario']==scenario] # subset dataframe to scenario
    # calculate relative change in coal combustion between 2010 and peak year
    peak_dict  = get_peak(df, var='Coal Combustion', yr_range=[2010, 2300])
    peak_value = peak_dict['peak_value']
    peak_year  = peak_dict['peak_year']
    base_value = df[df['Year']==2010]['Coal Combustion'].item()
    rel_change = get_relative_difference(base_value, peak_value)

    return scenario, peak_value, rel_change, peak_year

def get_select_year_vals(df, year:int, scenario:str):
    df = df[df['Scenario']==scenario] # subset dataframe to scenario
    # calculate relative change in coal combustion between 2010 and peak year
    select_value = df[df['Year']==year]['Coal Combustion'].item()
    base_value = df[df['Year']==2010]['Coal Combustion'].item()
    rel_change = get_relative_difference(base_value, select_value)

    return scenario, select_value, rel_change, peak_year

print('-- Displaying coal combustion and relative changes  ')
# read category emission data
df = pd.read_csv('../data/emissions_tabular/Category_Emission_Trends.csv')
scenarios = ['SSP1-26','SSP2-45','SSP5-34','SSP5-85']

print('--------------------------------------------------------------')
print('Scenario | Peak value (Mg/a) | Rel. change (%) | Peak year')
print('--------------------------------------------------------------')
for scenario in scenarios:
    scenario, peak_value, rel_change, peak_year = get_peak_vals(df, scenario)
    print(f'{scenario}'.ljust(5), ' |', 
          f'{peak_value}'.ljust(16), ' |', 
          f'{np.round(rel_change*100,1)}%'.ljust(14), ' |', 
          f'{peak_year}' )
    
# show 2050 values
print('--------------------------------------------------------------')
print('Scenario | 2050 value (Mg/a) | Rel. change (%) |')
print('--------------------------------------------------------------')
for scenario in scenarios:
    scenario, select_value, rel_change, peak_year = get_select_year_vals(df, 2050, scenario)
    print(f'{scenario}'.ljust(5), ' |', 
          f'{select_value}'.ljust(16), ' |', 
          f'{np.round(rel_change*100,1)}%'.ljust(14))
    
# show cumulative coal emissions
print('--------------------------------------------------------------')
print('Scenario | Cumulative coal emissions to air (Gg) |')
print('--------------------------------------------------------------')
cum_coal = {}
for scenario in scenarios:
    cumulative_coal = get_cumulative_emissions(df[df['Scenario']==scenario], var='Coal Combustion')
    cum_coal[scenario] = cumulative_coal
    print(f'{scenario}'.ljust(5), ' |', 
          f'{np.round(cumulative_coal,1)}'.ljust(16))

print('--------------------------------------------------------------')
# calculate factor difference between SSP5-85 and other scenarios
ratios = []
for scenario in ['SSP1-26','SSP2-45','SSP5-34']:
    ratio = cum_coal['SSP5-85']/cum_coal[scenario]
    ratios.append(ratio)

ratio_min, ratio_max = min(ratios), max(ratios)
print(' ')
print(f'Cumulative coal emissions under SSP5-8.5 are {np.round(ratio_min,1)}x to {np.round(ratio_max,1)}x higher than other scenarios')
print(' ')
# calculate fraction of total air emissions from coal
print('--------------------------------------------------------------')
print('Fraction of cumulative air emissions from coal (%) :')
print('--------------------------------------------------------------')
for scenario in scenarios:
    cum_coal = get_cumulative_emissions(df[df['Scenario']==scenario], var='Coal Combustion')
    cum_total = get_cumulative_emissions(df[df['Scenario']==scenario], var='Total')
    fraction = np.round(cum_coal/cum_total*100,1)
    print(f'{scenario}'.ljust(5), ' |', 
          f'{fraction}%'.ljust(16))
    
print(' ')
print('--------------------------------------------------------------')
print('--- 3.1.4 Changes in Hg emission speciation ')
print('--------------------------------------------------------------')

print('Fraction of air emissions as Hg0 (%) :')
speciation_out = pd.DataFrame()
for scenario in scenarios:
    df = pd.read_csv(f'../data/emissions_tabular/{scenario}.csv')
    df['Year'] = df['Year'].astype(int)
    df = df[(df['Year']>=2010) & (df['Year']<=2250)]
    df[scenario] = df['GLO_air_Hg0']/df['GLO_air_total']

    if len(speciation_out)==0:
        speciation_out = df[['Year', scenario]]
    else:
        speciation_out = pd.merge(speciation_out, df[['Year', scenario]], on='Year')

print(speciation_out.round(2))

print(' ')
print('--------------------------------------------------------------')
print('--- 3.2.1 Near-term deposition increases in some regions followed by longer-term declines ')
print('--------------------------------------------------------------')
print(' ')

df = pd.read_csv('../data/output/regional_deposition_attribution.csv')

def get_dep(df=df, scenario=str, year=int, source_category=str, receptor_regions=list):
    return df[(df['scenario']==scenario) & 
              (df['year']==year) & 
              (df['source_category']==source_category) &
              (df['receptor_region'].isin(receptor_regions))]['source_attributable_deposition'].sum()

#get_dep(df=df, scenario='SSP1-26', year=2010, source_category='legacy and natural', receptor_regions=['AfricaMidEast','Asia','Europe','NorthAm','Oceania','SouthAm','FmrUSSR'])

receptor_regions = ['AfricaMidEast','Asia','Europe','NorthAm','Oceania','SouthAm','FmrUSSR', 'OCEAN', 'OTHER']
anthro_receptors = ['AfricaMidEast','Asia','Europe','FmrUSSR','NorthAm','Oceania','SouthAm']
# delta primary (all-anthro regions)
ds = {'scenario':[], 'year':[], 'primary_dep':[], 'legacy_dep':[], 'total_dep':[], 'receptors':[]}
for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
    for year in np.arange(2010, 2110, 10):
        primary_dep = get_dep(df, scenario=scenario, year=year, source_category='primary', receptor_regions=anthro_receptors)
        legacy_dep = get_dep(df, scenario=scenario, year=year, source_category='legacy and natural', receptor_regions=anthro_receptors)
        total_dep = primary_dep+legacy_dep
        ds['scenario'].append(scenario)
        ds['year'].append(year)
        ds['primary_dep'].append(primary_dep)
        ds['legacy_dep'].append(legacy_dep)
        ds['total_dep'].append(total_dep)
        ds['receptors'].append(anthro_receptors)

ds = pd.DataFrame(ds)
# display total, legacy, and primary deposition to land in 2010
print(' ')
print('-- Total, primary, and legacy deposition to land in 2010')
print(' ')
tmp = ds[(ds['year']==2010) & (ds['scenario']=='SSP1-26')].copy() # note scenario doesn't matter here
print(f'Total   (2010) : {np.round(tmp["total_dep"].item(),1)} Mg/a')
print(f'Primary (2010) : {np.round(tmp["primary_dep"].item(),1)} Mg/a')
print(f'Legacy  (2010) : {np.round(tmp["legacy_dep"].item(),1)} Mg/a')
print(' ')

def get_delta(ds, t0, t, scenario='SSP5-85', dep_var='total_dep', year_var='year'):
    ds = ds[ds['scenario']==scenario]
    d0 = ds[ds[year_var]==t0][dep_var].values.item()
    d  = ds[ds[year_var]==t][dep_var].values
    delta = d - d0
    return delta

out_dict = {}
print('-- Change in source-type and magnitude of deposition to land')
for dep_var in ['total_dep', 'primary_dep', 'legacy_dep', 'f_legacy_dep']:
    print()
    output = pd.DataFrame()
    for t in [2020, 2030, 2050, 2100]:
        row = {'year':[t],'dep_var':[dep_var]}
        for scenario in ['SSP1-26', 'SSP2-45', 'SSP5-34', 'SSP5-85']:
            if dep_var == 'f_legacy_dep':
                delta_legacy = get_delta(ds=ds, t0=2010, t=t, scenario=scenario, dep_var='legacy_dep')
                delta_total  = get_delta(ds=ds, t0=2010, t=t, scenario=scenario, dep_var='total_dep')
                delta = delta_legacy/delta_total
            elif dep_var in ['total_dep', 'primary_dep', 'legacy_dep']:
                delta = get_delta(ds=ds, t0=2010, t=t, scenario=scenario, dep_var=dep_var)
            row[scenario] = delta

        output = pd.concat((output, pd.DataFrame(row)))
    print(' ')
    print(output)
    out_dict[dep_var] = output

# --------------------------------------------------------------
scenario = 'SSP1-26'
print(f'-- {scenario}')
dep_2010 = ds[(ds['scenario']==scenario) & (ds['year']==2010)]['total_dep'].values.item()
dep_2020 = ds[(ds['scenario']==scenario) & (ds['year']==2020)]['total_dep'].values.item()
print(f'Change in land deposition 2010 to 2020 under {scenario}: {np.round(dep_2020-dep_2010,1)} Mg/a ({np.round((dep_2020-dep_2010)/dep_2010*100,1)}%)')

dep_2100 = ds[(ds['scenario']==scenario) & (ds['year']==2100)]['total_dep'].values.item()
print(f'Deposition to land in 2100 under {scenario}: {np.round(dep_2100,1)} Mg/a ({np.round(dep_2100/dep_2010*100,1)}% of 2010)')

print(' ')
# calculate change in deposition over Asia (2010 - 2100)
primary_dep_Asia_2010 = get_dep(df, scenario=scenario, year=2010, source_category='primary', receptor_regions=['Asia'])
legacy_dep_Asia_2010  = get_dep(df, scenario=scenario, year=2010, source_category='legacy and natural', receptor_regions=['Asia'])
total_dep_Asia_2010   = primary_dep_Asia_2010+legacy_dep_Asia_2010
primary_dep_Asia_2100 = get_dep(df, scenario=scenario, year=2100, source_category='primary', receptor_regions=['Asia'])
legacy_dep_Asia_2100  = get_dep(df, scenario=scenario, year=2100, source_category='legacy and natural', receptor_regions=['Asia'])
total_dep_Asia_2100   = primary_dep_Asia_2100+legacy_dep_Asia_2100
print(f'Change in land deposition over Asia 2010 to 2100 under SSP1-2.6: {np.round(total_dep_Asia_2100-total_dep_Asia_2010,1)} Mg/a ({np.round((total_dep_Asia_2100-total_dep_Asia_2010)/total_dep_Asia_2010*100,1)}%)')

print(' ')
scenario = 'SSP2-45'
print(f'-- {scenario}')
dep_2010 = ds[(ds['scenario']==scenario) & (ds['year']==2010)]['total_dep'].values.item()
dep_2020 = ds[(ds['scenario']==scenario) & (ds['year']==2020)]['total_dep'].values.item()
dep_2030 = ds[(ds['scenario']==scenario) & (ds['year']==2030)]['total_dep'].values.item()
dep_2100 = ds[(ds['scenario']==scenario) & (ds['year']==2100)]['total_dep'].values.item()
print(f'Change in deposition to land 2010 to 2020 under {scenario}: {np.round(dep_2020-dep_2010,1)} Mg/a ({np.round((dep_2020-dep_2010)/dep_2010*100,1)}%)')
print(f'Change in deposition to land 2010 to 2030 under {scenario}: {np.round(dep_2030-dep_2010,1)} Mg/a ({np.round((dep_2030-dep_2010)/dep_2010*100,1)}%)')
print(f'Change in deposition to land 2010 to 2100 under {scenario}: {np.round(dep_2100-dep_2010,1)} Mg/a ({np.round((dep_2100-dep_2010)/dep_2010*100,1)}%)')
print(' ')
regional_dep_trends = pd.read_csv('../data/output/regional_deposition_trends.csv')
Europe_2020 = regional_dep_trends[(regional_dep_trends['scenario']==scenario) & (regional_dep_trends['year']==2020) & (regional_dep_trends['receptor region']=='Europe')]
Europe_2030 = regional_dep_trends[(regional_dep_trends['scenario']==scenario) & (regional_dep_trends['year']==2030) & (regional_dep_trends['receptor region']=='Europe')]
print(f'Deposition change over Europe 2010 to 2020: {np.round(Europe_2020["change since 2010 [%]"].values.item(),1)}%')
print(f'Deposition change over Europe 2010 to 2030: {np.round(Europe_2030["change since 2010 [%]"].values.item(),1)}%')
Asia_2010 = regional_dep_trends[(regional_dep_trends['scenario']==scenario) & (regional_dep_trends['year']==2010) & (regional_dep_trends['receptor region']=='Asia')]
Asia_2100 = regional_dep_trends[(regional_dep_trends['scenario']==scenario) & (regional_dep_trends['year']==2100) & (regional_dep_trends['receptor region']=='Asia')]

print(f'Deposition change over Asia 2010 to 2100: {np.round((Asia_2100["deposition [ug/m2/yr]"].values.item()-Asia_2010["deposition [ug/m2/yr]"].values.item()), 1)} ug/m2/yr ({np.round(Asia_2100["change since 2010 [%]"].values.item(),1)}%)')

def get_peak(df, var:str, yr_range=[2010, 2300]):
    '''Calculate the peak value of a variable between a start year and an end year
        df: dataframe
        var: variable to calculate peak in
        yr_range: range of years to calculate peak in
        
        returns: dictionary with peak year and peak value'''
    
    df = df[(df['Year']>=yr_range[0]) & (df['Year']<=yr_range[1])]
    peak_year = df[df[var]==df[var].max()]['Year'].item()
    peak_value = df[var].max()
    return {'peak_year':peak_year, 'peak_value':peak_value}

print(' ')
scenario = 'SSP5-34'
print(f'-- {scenario}')
# get year and value of highest deposition
tmp = ds[ds['scenario']==scenario]
dep_2010   = tmp[(tmp['year']==2010)]['total_dep'].values.item()
peak_year  = tmp[tmp['total_dep']==tmp['total_dep'].max()]['year'].item()
peak_value = tmp[(tmp['year']==peak_year)]['total_dep'].values.item()
print(f'Peak land deposition under {scenario}: {np.round(peak_value,1)} Mg/a ({np.round(((peak_value/dep_2010)-1)*100,1)}% greater than 2010)')

# get increase in deposition over Asia
Asia_2010 = regional_dep_trends[(regional_dep_trends['scenario']==scenario) & (regional_dep_trends['year']==2010) & (regional_dep_trends['receptor region']=='Asia')]
Asia_2030 = regional_dep_trends[(regional_dep_trends['scenario']==scenario) & (regional_dep_trends['year']==2030) & (regional_dep_trends['receptor region']=='Asia')]
print(f'Peak deposition in Asia: {np.round((Asia_2030["deposition [ug/m2/yr]"].values.item()), 1)} ug/m2/yr ({np.round(Asia_2030["change since 2010 [%]"].values.item(),1)}% greater than 2010)')

dep_2100 = tmp[(tmp['year']==2100)]['total_dep'].values.item()
print(f'Deposition to land in 2100 under {scenario}: {np.round(dep_2100,1)} Mg/a (change from 2010 to 2100: {np.round(dep_2100 - dep_2010,1)} Mg/a)')

print(' ')
scenario = 'SSP5-85'
print(f'-- {scenario}')
Africa_2010 = regional_dep_trends[(regional_dep_trends['scenario']==scenario) & (regional_dep_trends['year']==2010) & (regional_dep_trends['receptor region']=='AfricaMidEast')]['deposition [ug/m2/yr]'].values.item()
# get year and value of highest deposition in Africa and Middle East from regional_dep_trends
tmp = regional_dep_trends[(regional_dep_trends['scenario']==scenario) & (regional_dep_trends['receptor region']=='AfricaMidEast')]
dep_2010   = tmp[(tmp['year']==2010)]['deposition [ug/m2/yr]'].values.item()
peak_year  = tmp[tmp['deposition [ug/m2/yr]']==tmp['deposition [ug/m2/yr]'].max()]['year'].item()
peak_value = tmp[(tmp['year']==peak_year)]['deposition [ug/m2/yr]'].values.item()
print(f'Deposition increase over Africa and Middle East: {np.round(peak_value-dep_2010,1)} ug/m2/yr by {peak_year} ({np.round(((peak_value/dep_2010)-1)*100,1)}% above 2010)')
# get end of century deposition
dep_2010 = ds[(ds['scenario']==scenario) & (ds['year']==2010)]['total_dep'].values.item()
dep_2100 = ds[(ds['scenario']==scenario) & (ds['year']==2100)]['total_dep'].values.item()
print(f'Change in deposition to land 2010 - 2100 under {scenario}: {np.round((dep_2100 - dep_2010),1)} Mg/a')

print(' ')
print('--------------------------------------------------------------')
print('--- 3.3. Implications for Legacy Hg Deposition from Terrestrial and Aquatic Emissions ')
print('--------------------------------------------------------------')
print(' ')

df = pd.read_csv('../data/output/annual_legacy_deposition_to_land.csv')

def get_peak_metrics(df:pd.DataFrame, column:str, base_year:int=2010):
    idx_max    = df[column].idxmax()
    value_max  = df[column].loc[idx_max]
    yr_max     = df['year'].loc[idx_max]
    value_base = df[column].loc[df['year']==base_year].values[0]
    delta = value_max-value_base
    return yr_max, value_max, delta

print('-- Displaying legacy deposition to land by scenario')
base_year = 2010
print('--------------------------------------------------------------')
print(f'Scenario | Peak Dep. (Mg/a) | Peak Year | Change from {int(base_year)} (Mg/a)')
print('--------------------------------------------------------------')
for scenario in scenarios:
    peak_year, peak_deposition, delta = get_peak_metrics(df, scenario, base_year=base_year)
    print(f'{scenario}'.ljust(5), ' |', 
          f'{np.round(peak_deposition,1)}'.ljust(15), ' | ', 
          f'{int(peak_year)}'.ljust(7), ' |', 
          f'{np.round(delta,1)}'.ljust(5), '')
    
def get_dep_change(df, column, target_year:int=2100, base_year:int=2010):
    value_base = df[column].loc[df['year']==base_year].values[0]
    value_year = df[column].loc[df['year']==target_year].values[0]
    delta = value_year-value_base
    return value_year, delta

print(' ')
print('-- Displaying end-century (2100) legacy deposition to land by scenario')
print('--------------------------------------------------------------')
print(f'Scenario | 2100 Dep. (Mg/a) | Change from {base_year} (Mg/a)')
print('--------------------------------------------------------------')

for scenario in scenarios:
    value_year, delta = get_dep_change(df, scenario, base_year=base_year)
    print(f'{scenario}'.ljust(5), ' |', 
          f'{np.round(value_year,1)}'.ljust(15), ' | ', 
          f'{np.round(delta,1)}'.ljust(5), '')
    
# -- regional source-receptor relationships for legacy emissions
print(' ')
print('-- Displaying (relative) regional source-receptor relationships for legacy emissions')
print(' ')

label_dict = {'AfricaMidEast':'AFM', 'Asia':'ASA', 'Europe':'EUR', 'FmrUSSR':'FSU', 
              'NorthAm':'NAM', 'Oceania':'OCA', 'SouthAm':'SAM',
              'OCEAN':'OCE', 'OTHER':'OTH', 'E_LAND':'Soil', 'E_OCEAN':'Ocean'}

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

    df['Rel. Dep.'] = df['areal_deposition']/average # calculate relative deposition
    return df

df = get_relative_deposition(emission_category='E_LAND')
print('Area-normalized Deposition from Land Emissions (relative to area-weighted average):')
print(df[['Receptor', 'Rel. Dep.']].round(2))
print(' ')

df = get_relative_deposition(emission_category='E_OCEAN')
print('Area-normalized Deposition from Ocean Emissions (relative to area-weighted average):')
print(df[['Receptor', 'Rel. Dep.']].round(2))

# --- Display seawater Hg trends 
def get_surfint_conc(surfint_Hg_mass:float, oci_depth_km:float):
    # -- note: future applications should consider using volume fractions calculated 
    #    directly from ETOPO bathymetry, rather than assuming that surfintvol can be
    #    calculated from ocean surface area
    ocean_surface_area = 3.619e8 # km2
    km3_to_L           = 1e12    # km3 -> L
    kg_to_mol          = 1e3/200.59 # kg Hg -> mol Hg
    surfintvol = (ocean_surface_area*oci_depth_km)*km3_to_L # surface + intermediate ocean volume in L
    conc_intoc  = surfint_Hg_mass*1e3*kg_to_mol*1e12/surfintvol # concentration in pM
    return conc_intoc # concentration in pM

output = pd.DataFrame(columns=['Scenario', 'Peak Year', 'Peak Mass (Gg)', 'Change 2010-peak (Gg)', 'Change 2010-peak (%)', '2100 Mass (Gg)', 'Change 2010-2100 (Gg)', 'Change 2010-2100 (%)'])
# read in data
df = pd.read_csv('../data/box_model/output/reservoir_trends.csv')
df['upper ocean'] = df[['ocs','oci']].sum(axis=1)
print(' ')
print('Displaying PEAK YEAR upper ocean (0-1500m) Hg mass and concentration')
print(' ')
Mg_to_Gg = 1e-3
for scenario in df['scenario'].unique():
    sel = df[df['scenario']==scenario].copy()
    base_value = sel[sel['Year']==2010]['upper ocean'].item()
    end_century_value = sel[sel['Year']==2100]['upper ocean'].item()
    peak_value = sel['upper ocean'].max()
    peak_year = sel[sel['upper ocean']==peak_value]['Year'].item()
    delta = get_delta(df, 2010, peak_year, scenario=scenario, dep_var='upper ocean', year_var='Year').item() # change since 2010 (Gg)
    delta2 = get_delta(df, 2010, 2100, scenario=scenario, dep_var='upper ocean', year_var='Year').item() # change since 2010 (Gg)
    rel_delta_pct = (delta/base_value)*100 # percent change from base to peak
    rel_delta2_pct = (delta2/base_value)*100 # percent change from base to peak
    conc_base_pM = get_surfint_conc(base_value, oci_depth_km=1.5)
    conc_peak_pM = get_surfint_conc(peak_value, oci_depth_km=1.5)
    conc_endcentury_pM = get_surfint_conc(end_century_value, oci_depth_km=1.5)

    tmp = pd.DataFrame({'Scenario':scenario, 
                        'Peak Year':peak_year, 
                        'Peak Mass (Gg)':np.round(peak_value*Mg_to_Gg,1), 
                        'Peak Concentration (pM)': np.round(conc_peak_pM,1),
                        'Change 2010-peak (Gg)':np.round(delta*Mg_to_Gg,1), 
                        'Change 2010-peak (pM)':np.round(conc_peak_pM-conc_base_pM,2),
                        'Change 2010-peak (%)':np.round(rel_delta_pct,1), 
                        '2100 Mass (Gg)':np.round(end_century_value*Mg_to_Gg,1),
                        '2100 Concentration (pM)':np.round(conc_endcentury_pM,1),
                        'Change 2010-2100 (Gg)':np.round(delta2*Mg_to_Gg, 1),
                        'Change 2010-2100 (pM)':np.round(conc_endcentury_pM-conc_base_pM,2),
                        'Change 2010-2100 (%)':np.round(rel_delta2_pct,1)}, index=[0])
    output = pd.concat([output, tmp], axis=0)

output = output.reset_index(drop=True)
print(output[['Scenario', 'Peak Year', 'Peak Mass (Gg)', 'Peak Concentration (pM)']])
print(' ')
print('Displaying change in upper ocean Hg mass and concentration from 2010 to peak')
print(' ')
print(output[['Scenario', 'Change 2010-peak (Gg)', 'Change 2010-peak (pM)', 'Change 2010-peak (%)']])
print(' ')
print('Displaying END CENTURY (2100) upper ocean (0-1500m) Hg mass and concentration')
print(' ')
print(output[['Scenario', '2100 Mass (Gg)', '2100 Concentration (pM)']])
print(' ')
print('Displaying change in upper ocean Hg mass and concentration from 2010 to 2100')
print(' ')
print(output[['Scenario', 'Change 2010-2100 (Gg)', 'Change 2010-2100 (pM)', 'Change 2010-2100 (%)']])

# -- 