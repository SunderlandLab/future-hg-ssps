# future-hg-ssps

<a href="https://doi.org/10.5281/zenodo.10672520"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10672520.svg" alt="DOI"></a>

## Introduction

Emission files, model output, and supporting data are stored separately on <a href="https://doi.org/10.7910/DVN/UIEZW5">Harvard Dataverse</a> and can be downloaded to the repository directory using `fetch_dataverse_files.py`.

The script `call_all.sh` executes the python files listed below. Dependencies for python are listed in `requirements.txt`.

Source code for the paper:
> Geyman, B.M., Streets, D.G., Thackray, C.P. Olson, C.L., Schaefer, K., and Sunderland, E.M. (2024). Projecting Global Mercury Emissions and Deposition Under the Shared Socioeconomic Pathways. *ESS Open Archive*. <a href="https://doi.org/10.22541/essoar.169945526.69817769/v1">https://doi.org/10.22541/essoar.169945526.69817769/v1</a>

## Authors
* Benjamin M. Geyman
* David G. Streets
* Colin P. Thackray
* Christine L. Olson
* Kevin Schaefer
* [Elsie M. Sunderland](https://bgc.seas.harvard.edu/)

##  Order and purpose of scripts

Main analysis may be run by calling `./scripts/call_all.sh`

### Main Scripts
- fetch_dataverse_files.py
- attribute_regional_deposition.py
- plot_1_regional_and_category_emissions.py
- plot_2_emission_speciation_trends.py
- plot_3_relative_regional_deposition_change.py
- plot_4_global_deposition_maps.py
- plot_5_legacy_deposition_trends.py
- plot_6_area_normalized_legacy_deposition.py
- plot_S1_compare_to_recent_emissions.py
- plot_S2_own_source_dep.py
- plot_S3_legacy_vs_primary_dep.py
- plot_S4_legacy_deposition_change_map.py
- plot_S5_upper_ocean_mass_trends.py
- plot_S6_source_receptor_matrices.py
- call_all.sh
